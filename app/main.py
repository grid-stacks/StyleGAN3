import nvsmi
import torch
import sys

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="StyleGAN3 API")


@app.get("/")
def home():
    return {"hello": "world"}


@app.get("/gpu_type")
def gpu_type():
    return {"gpus": list(nvsmi.get_gpus()), "available_gpus": list(nvsmi.get_available_gpus()),
            "gpu_processes": nvsmi.get_gpu_processes()}


@app.get("/torch_device")
def torch_device():
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)
    return {"device": str(device)}
