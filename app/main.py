from fastapi import FastAPI

app = FastAPI(title="StyleGAN3 API")


@app.get("/")
def home():
    return {"hello": "world"}
