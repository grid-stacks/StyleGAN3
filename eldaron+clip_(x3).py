# -*- coding: utf-8 -*-
"""Eldaron+Clip (x3).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vu-1F2y6fTB-83DPajSAy6TY_XMti0lj

# **StyleGAN3 + CLIP 🖼️**

## Generate images from text prompts using NVIDIA's StyleGAN3 with CLIP guidance.

Head over [here](https://github.com/ouhenio/StyleGAN3-CLIP-notebook) if you want to be up to date with the changes to this notebook and play with other alternatives.

The original code was written by [nshepperd](https://twitter.com/nshepperd1)* (https://github.com/nshepperd), and later edited by [Eugenio Herrera](https://twitter.com/ouhenio) (https://github.com/ouhenio).

Thanks to [Katherine Crowson](https://twitter.com/RiversHaveWings) (https://github.com/crowsonkb) for coming up with many improved sampling tricks, as well as some of the code.

----

(*) nshepperd originally made [this notebook](https://colab.research.google.com/drive/1eYlenR1GHPZXt-YuvXabzO9wfh9CWY36#scrollTo=LQf7tzBQ8rn2).

(**) The interface is inspired by [this notebook](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP(Updated).ipynb), done by Jakeukalane and Avengium (Angel).

(***) For more information about StyleGAN3, [visit the official repository](https://github.com/NVlabs/stylegan3).
"""

#@markdown #**Check GPU type** 🕵️
#@markdown ### Factory reset runtime if you don't have the desired GPU.

#@markdown ---




#@markdown V100 = Excellent (*Available only for Colab Pro users*)

#@markdown P100 = Very Good

#@markdown T4 = Good (*preferred*)

#@markdown K80 = Meh

#@markdown P4 = (*Not Recommended*) 

#@markdown ---

!nvidia-smi -L

#@markdown #**Install libraries** 🏗️
# @markdown This cell will take a little while because it has to download several libraries.

#@markdown ---

!pip install --upgrade torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install --upgrade https://download.pytorch.org/whl/nightly/cu111/torch-1.11.0.dev20211012%2Bcu111-cp37-cp37m-linux_x86_64.whl https://download.pytorch.org/whl/nightly/cu111/torchvision-0.12.0.dev20211012%2Bcu111-cp37-cp37m-linux_x86_64.whl
!git clone https://github.com/NVlabs/stylegan3
!git clone https://github.com/openai/CLIP
!pip install -e ./CLIP
!pip install einops ninja
!pip install --upgrade --no-cache-dir gdown

import sys
sys.path.append('./CLIP')
sys.path.append('./stylegan3')

import io
import os, time, glob
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
from google.colab import files

device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

#@markdown #**Optional:** Save images in Google Drive 💾
# @markdown Run this cell if you want to store the results inside Google Drive.

# @markdown Copying the generated images to drive is faster to work with.

# @markdown **Important**: you must have a folder named *samples* inside your drive, otherwise this may not work.

#@markdown ---

# Uncomment to copy generated images to drive, faster than downloading directly from colab in my experience.
from google.colab import drive
drive.mount('/content/drive')

#@markdown #**Define necessary functions** 🛠️

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
          !wget -c '{url_or_path}'
        else:
          path_id = url_or_path.split("id=")[-1]
          !gdown --id '{path_id}'
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
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
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
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

make_cutouts = MakeCutouts(224, 32, 0.5)

def embed_image(image):
  n = image.shape[0]
  cutouts = make_cutouts(image)
  embeds = clip_model.embed_cutout(cutouts)
  embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
  return embeds

def embed_url(url):
  image = Image.open(fetch(url)).convert('RGB')
  return embed_image(TF.to_tensor(image).to(device).unsqueeze(0)).mean(0).squeeze(0)

class CLIP(object):
  def __init__(self):
    clip_model = "ViT-B/32"
    self.model, _ = clip.load(clip_model)
    self.model = self.model.requires_grad_(False)
    self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                          std=[0.26862954, 0.26130258, 0.27577711])

  @torch.no_grad()
  def embed_text(self, prompt):
      "Normalized clip text embedding."
      return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

  def embed_cutout(self, image):
      "Normalized clip image embedding."
      return norm1(self.model.encode_image(self.normalize(image)))
  
clip_model = CLIP()

#@markdown #**Model selection** 🎭


#@markdown There are 4 pre-trained options to play with:
#@markdown - FFHQ: Trained with human faces.
#@markdown - MetFaces: Trained with paintings/portraits of human faces.
#@markdown - AFHQv2: Trained with animal faces.
#@markdown - Cosplay: Trained by [l4rz](https://twitter.com/l4rz) with cosplayer's faces.
#@markdown - Wikiart: Trained by [Justin Pinkney](https://www.justinpinkney.com/) with the Wikiart 1024 dataset.
#@markdown - Landscapes: Trained by [Justin Pinkney](https://www.justinpinkney.com/) with the LHQ dataset.


#@markdown **Run this cell again if you change the model**.

#@markdown ---

base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"

Model = 'ELDARON' #@param ["FFHQ", "MetFaces", "AFHQv2", "cosplay", "Wikiart", "Landscapes", "ELDARON"]

#@markdown ---

model_name = {
    "FFHQ": base_url + "stylegan3-t-ffhqu-1024x1024.pkl",
    "MetFaces": base_url + "stylegan3-r-metfacesu-1024x1024.pkl",
    "AFHQv2": base_url + "stylegan3-t-afhqv2-512x512.pkl",
    "cosplay": "https://l4rz.net/cosplayface-snapshot-stylegan3t-008000.pkl",
    "Wikiart": "https://drive.google.com/u/0/open?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj",
    "Landscapes": "https://drive.google.com/u/0/open?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1",
    "ELDARON": "https://drive.google.com/u/0/uc?id=1dKUy4q5HtvcfiLCwj9nEITgVArBbEGIo"
}

network_url = model_name[Model]

with open(fetch_model(network_url), 'rb') as fp:
  G = pickle.load(fp)['G_ema'].to(device)

zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)

#@markdown #**Parameters** ✍️
#@markdown `texts`: Enter here a prompt to guide the image generation. You can enter more than one prompt separated with
#@markdown `|`, which will cause the guidance to focus on the different prompts at the same time, allowing to mix and play
#@markdown with the generation process.

#@markdown `steps`: Number of optimization steps. The more steps, the longer it will try to generate an image relevant to the prompt.

#@markdown `seed`: Determines the randomness seed. Using the same seed and prompt should give you similar results at every run.
#@markdown Use `-1` for a random seed.

#@markdown ---

texts = "Portrait"#@param {type:"string"}
steps = 1#@param {type:"number"}
seed_1 = 20#@param {type:"number"}
seed_2 = 23#@param {type:"number"}
seed_3 = 24#@param {type:"number"}

seeds = [seed_1, seed_2, seed_3]

#@markdown ---

if seed_1 == -1:
    seed = np.random.randint(0,9e9)
    print(f"Your random seed is: {seed}")

texts = [frase.strip() for frase in texts.split("|") if frase]

targets = [clip_model.embed_text(text) for text in texts]

#@markdown #**Run the model** 🚀

# Actually do the run

tf = Compose([
  Resize(224),
  lambda x: torch.clamp((x+1)/2,min=0,max=1),
  ])

def run(timestring, seed):
  torch.manual_seed(seed)

  # Init
  # Sample 32 inits and choose the one closest to prompt

  with torch.no_grad():
    qs = []
    losses = []
    for _ in range(8):
      q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
      images = G.synthesis(q * w_stds + G.mapping.w_avg)
      embeds = embed_image(images.add(1).div(2))
      loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
      i = torch.argmin(loss)
      qs.append(q[i])
      losses.append(loss[i])
    qs = torch.stack(qs)
    losses = torch.stack(losses)
    # print(losses)
    # print(losses.shape, qs.shape)
    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0).requires_grad_()

  # Sampling loop
  q_ema = q
  opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0,0.999))
  loop = tqdm(range(steps))
  for i in loop:
    opt.zero_grad()
    w = q * w_stds
    image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
    embed = embed_image(image.add(1).div(2))
    loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
    loss.backward()
    opt.step()
    loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

    q_ema = q_ema * 0.9 + q * 0.1
    image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, noise_mode='const')

    if i % 10 == 0:
      display(TF.to_pil_image(tf(image)[0]))
      print(f"Image {i}/{steps} | Current loss: {loss}")
    pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
    os.makedirs(f'samples/{timestring}', exist_ok=True)
    pil_image.save(f'samples/{timestring}/{i:04}.jpg')

try:
  timestrings = []
  for seed in seeds:
    timestring = time.strftime('%Y%m%d%H%M%S')
    timestrings.append(timestring)
    run(timestring, seed)
except KeyboardInterrupt:
  pass

#@markdown #**Save images** 📷
#@markdown A `.tar` file will be saved inside *samples* and automatically downloaded, unless you previously ran the Google Drive cell,
#@markdown in which case it'll be saved inside your previously created drive *samples* folder.

archive_name = "Druid. Fantasy Art. 20 21 22"#@param {type:"string"}

archive_name = slugify(archive_name)

if archive_name != "optional":
  fname = archive_name
  # os.rename(f'samples/{timestring}', f'samples/{fname}')
else:
  fname = timestring
# Save images as a tar archive
for fname in timestrings:
  !tar cf samples/{fname}.tar samples/{timestring}
  if os.path.isdir('drive/MyDrive/samples'):
    shutil.copyfile(f'samples/{fname}.tar', f'drive/MyDrive/samples/{fname}.tar')
  else:
    files.download(f'samples/{fname}.tar')

#@markdown #**Generate video** 🎥

#@markdown You can edit frame rate and stuff by double-clicking this tab.
for timestring in timestrings:

  frames = os.listdir(f"samples/{timestring}")
  frames = len(list(filter(lambda filename: filename.endswith(".jpg"), frames))) #Get number of jpg generated

  init_frame = 1 #This is the frame where the video will start
  last_frame = frames #You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

  min_fps = 10
  max_fps = 60

  total_frames = last_frame-init_frame

  #Desired video time in seconds
  video_length = 14 #@param {type:"number"}
  #Video filename
  video_name = "Portrait of Aldemar, Archmage of the Mist. Painting (c. 1583)" #@param {type:"string"}
  video_name = slugify(video_name)

  if not video_name:
    video_name = "video_" + timestring
  # frames = []
  # tqdm.write('Generating video...')
  # for i in range(init_frame,last_frame): #
  #     filename = f"samples/{timestring}/{i:04}.jpg"
  #     frames.append(Image.open(filename))

  fps = np.clip(total_frames/video_length,min_fps,max_fps)

  print("Generating video...")
  !ffmpeg -r {fps} -i samples/{timestring}/%04d.jpg -c:v libx264 -vf fps={fps} -pix_fmt yuv420p samples/{video_name}.mp4 -frames:v {total_frames}

  # from subprocess import Popen, PIPE
  # p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', f'samples/{video_name}.mp4'], stdin=PIPE)
  # for im in tqdm(frames):
  #     im.save(p.stdin, 'PNG')
  # p.stdin.close()

  print("The video is ready")

#@markdown #**Download video** 📀
#@markdown If you're activated the download to GDrive option, the video will be save there. Don't worry about overwritting issues for colliding filenames, an id will be added to them to avoid this.

#Video filename
to_download_video_name = "Andelar, Robed Mage of the Mist" #@param {type:"string"}
to_download_video_name = slugify(to_download_video_name)

if not to_download_video_name:
  to_download_video_name = "video"


from google.colab import files
for timestring in timestrings:

  if os.path.isdir('drive/MyDrive/samples'):
    filelist = glob.glob(f'drive/MyDrive/samples/{to_download_video_name}*.mp4')
    video_count = len(filelist)
    if video_count:
      final_video_name = f"{to_download_video_name}{video_count}"
    else:
      final_video_name = to_download_video_name
    shutil.copyfile(f'samples/{video_name}.mp4', f'drive/MyDrive/samples/{final_video_name}.mp4')
  else:
    files.download(f"samples/{to_download_video_name}.mp4")

import anvil.server

anvil.server.connect("E7KQ75TS5PII2C5KJQPXHXWM-MTHW4EZMK5UNQAU5")

@anvil.server.callable
def predict_iris(x):
  print(x)
  return 'Hi ' + x

anvil.server.wait_forever()

#@title Licensed under the MIT License { display-mode: "form" }

# Copyright (c) 2021 nshepperd; Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.