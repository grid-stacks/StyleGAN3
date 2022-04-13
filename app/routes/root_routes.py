import nvsmi
import torch
import sys

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {"hello": "world"}


@router.get("/gpu_type")
def gpu_type():
    return {"gpus": list(nvsmi.get_gpus()), "available_gpus": list(nvsmi.get_available_gpus()),
            "gpu_processes": nvsmi.get_gpu_processes()}


@router.get("/torch_device")
def torch_device():
    device = torch.device('cuda:0')
    print('Using device:', device, file=sys.stderr)
    return {"device": str(device)}
