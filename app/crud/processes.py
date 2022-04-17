import io
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata

import clip
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from IPython.display import display
from PIL import Image
from einops import rearrange
from fastapi import Body
from starlette.responses import FileResponse
from torchvision.transforms import Compose, Resize
from tqdm.notebook import tqdm

from app.crud.models import retrieve_model
from app.schemas import ModelOptionEnum, SeedSchema

gan3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'stylegan3'
)
# tu_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan3', 'torch_utils')
# dnnlib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan3', 'dnnlib')

sys.path.append("../stylegan3")
# sys.path.append(tu_path)
# sys.path.append(dnnlib_path)

device = torch.device('cuda:0')

tf = Compose([
    Resize(224),
    lambda x: torch.clamp((x + 1) / 2, min=0, max=1),
])


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if "drive.google" in url_or_path:
        if "18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj" in url_or_path:
            basename = "wikiart-1024-stylegan3-t-17.2Mimg.pkl"
        elif "14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1" in url_or_path:
            basename = "lhq-256-stylegan3-t-25Mimg.pkl"
        elif "1dKUy4q5HtvcfiLCwj9nEITgVArBbEGIo" in url_or_path:
            basename = "0409i.pkl"
    if os.path.exists(basename):
        return basename
    else:
        if "drive.google" not in url_or_path:
            # !wget -c '{url_or_path}'
            p = subprocess.Popen(f"wget -c '{url_or_path}'", stdout=subprocess.PIPE, shell=True)
            print(p.communicate())
        else:
            path_id = url_or_path.split("id=")[-1]
            # !gdown --id '{path_id}'
            p = subprocess.Popen(f"gdown --id '{path_id}'", stdout=subprocess.PIPE, shell=True)
            print(p.communicate())
        return basename


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def norm1(prompt):
    """Normalize to the unit sphere."""
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1:  # Keeps consitent results vs previous method for single objective guidance
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


make_cutouts = MakeCutouts(224, 32, 0.5)


class CLIP(object):
    def __init__(self):
        _clip_model = "ViT-B/32"
        self.model, _ = clip.load(_clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        """Normalized clip text embedding."""
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

    def embed_cutout(self, image):
        """Normalized clip image embedding."""
        return norm1(self.model.encode_image(self.normalize(image)))


clip_model = CLIP()


def embed_image(image):
    n = image.shape[0]
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds


def embed_url(url):
    image = Image.open(fetch(url)).convert('RGB')
    return embed_image(TF.to_tensor(image).to(device).unsqueeze(0)).mean(0).squeeze(0)


def G(model: ModelOptionEnum):
    print("--1")
    try:
        model = retrieve_model(model)
        print("--1 a")
        with open(model["file"], 'rb') as fp:
            print("--1 b")
            try:
                print("--1 c")
                g = pickle.load(fp)['G_ema'].to(device)
                print("--1 d")
                return g
            except Exception as e:
                print('-- 3', e)

    except ModuleNotFoundError as e:
        return {"error": str(e)}

    print("--1 z")


def w_stds(model: ModelOptionEnum):
    print("--2")
    try:
        g = G(model)
        print("--2 a")
        zs = torch.randn([10000, g.mapping.z_dim], device=device)
        print("--2 b")
        w = g.mapping(zs, None).std(0)
        print("--2 c")
        return g, w
    except Exception as e:
        print('--l199', e)

    print("--2 z")


def seeding(data: SeedSchema):
    print("--3")
    seeds = [data.seed_1, data.seed_2, data.seed_3]

    if data.seed_1 == -1:
        seed = np.random.randint(0, 9e9)
        print(f"Your random seed is: {seed}")

    texts = [frase.strip() for frase in data.texts.split("|") if frase]

    targets = [clip_model.embed_text(text) for text in texts]

    print("--3 z")

    return seeds, targets


def run(timestring, seed, model: ModelOptionEnum, targets, steps):
    print("--4")
    g, w = w_stds(model)
    print("--4 a")

    torch.manual_seed(seed)
    print("--4 b")

    # Init
    # Sample 32 inits and choose the one closest to prompt

    with torch.no_grad():
        print("--4 c")
        qs = []
        losses = []
        for _ in range(8):
            print("--4 d")
            q = (g.mapping(torch.randn([4, g.mapping.z_dim], device=device), None,
                           truncation_psi=0.7) - g.mapping.w_avg) / w
            print("--4 e")
            images = g.synthesis(q * w + g.mapping.w_avg)
            print("--4 f")
            embeds = embed_image(images.add(1).div(2))
            print("--4 g")
            loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
            print("--4 h")
            i = torch.argmin(loss)
            qs.append(q[i])
            losses.append(loss[i])
            print("--4 i")
        qs = torch.stack(qs)
        print("--4 j")
        losses = torch.stack(losses)
        print("--4 k")
        # print(losses)
        # print(losses.shape, qs.shape)
        i = torch.argmin(losses)
        q = qs[i].unsqueeze(0).requires_grad_()

    # Sampling loop
    q_ema = q
    opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0, 0.999))
    loop = tqdm(range(steps))
    for i in loop:
        opt.zero_grad()
        w = q * w
        image = g.synthesis(w + g.mapping.w_avg, noise_mode='const')
        embed = embed_image(image.add(1).div(2))
        loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
        loss.backward()
        opt.step()
        loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

        q_ema = q_ema * 0.9 + q * 0.1
        image = g.synthesis(q_ema * w + g.mapping.w_avg, noise_mode='const')

        if i % 10 == 0:
            display(TF.to_pil_image(tf(image)[0]))
            print(f"Image {i}/{steps} | Current loss: {loss}")
        pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1))
        os.makedirs(f'samples/{timestring}', exist_ok=True)
        pil_image.save(f'samples/{timestring}/{i:04}.jpg')
        print("--6")


def timestring_run(data: SeedSchema, model: ModelOptionEnum):
    seeds, targets = seeding(data)

    try:
        timestrings = []
        for seed in seeds:
            timestring = time.strftime('%Y%m%d%H%M%S')
            print("--7", timestring)
            timestrings.append(timestring)
            run(timestring, seed, model, targets, data.steps)

        return timestrings
    except KeyboardInterrupt:
        pass


def generate_image(data: SeedSchema, model: ModelOptionEnum, archive_name: str = Body(...)):
    # timestrings = timestring_run(data, model)

    folder = os.path.abspath("files/samples")

    archive_name = slugify(archive_name)

    if archive_name != "optional":
        fname = archive_name
        # os.rename(f'samples/{timestring}', f'samples/{fname}')
    else:
        fname = timestring

    # Save images as a tar archive
    for fname in timestrings:
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_name = f"{fname}.tar"
        file_path = os.path.abspath(f"{folder}/{file_name}")

        p = subprocess.Popen(f"!tar cf {file_path} {folder}/{timestring}", stdout=subprocess.PIPE, shell=True)
        print(p.communicate())

        return FileResponse(path=file_path, filename=file_name, media_type='application/x-tar')


def generate_video(data: SeedSchema, model: ModelOptionEnum, video_name: str = Body(...)):
    timestrings = timestring_run(data, model)

    folder = os.path.abspath("files/samples")

    for timestring in timestrings:
        frames = os.listdir(f"{folder}/{timestring}")
        frames = len(list(filter(lambda filename: filename.endswith(".jpg"), frames)))  # Get number of jpg generated

        init_frame = 1  # This is the frame where the video will start
        last_frame = frames  # You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

        min_fps = 10
        max_fps = 60

        total_frames = last_frame - init_frame

        # Desired video time in seconds
        video_length = 14  # @param {type:"number"}

        # Video filename
        video_name = slugify(video_name)

        if not video_name:
            video_name = "video_" + timestring

        fps = np.clip(total_frames / video_length, min_fps, max_fps)

        video_name = f"{video_name}.mp4"
        video_path = os.path.abspath(f"{folder}/{video_name}")
        image_path = os.path.abspath(f"{folder}/{timestring}/%04d.jpg")

        print("Generating video...")
        cmd = f"ffmpeg -r {fps} -i {image_path} -c:v libx264 -vf fps={fps} -pix_fmt yuv420p {video_path} -frames:v {total_frames}"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        print(p.communicate())

        print("The video is ready")

        FileResponse(path=video_path, filename=video_name, media_type='video/mp4')

    return {"msg": "Video downloaded"}
