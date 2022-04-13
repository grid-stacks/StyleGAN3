import nvsmi

from fastapi import FastAPI

app = FastAPI(title="StyleGAN3 API")


@app.get("/")
def home():
    return {"hello": "world"}


@app.get("/gpu_type")
def gpu_type():
    return {"gpus": list(nvsmi.get_gpus()), "available_gpus": list(nvsmi.get_available_gpus()),
            "gpu_processes": nvsmi.get_gpu_processes()}
