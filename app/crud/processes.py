import sys

import io
import os
import time
import glob
import subprocess
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
import unicodedata
import re
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
from einops import rearrange

device = torch.device('cuda:0')

sys.path.append('../stylegan3')


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
