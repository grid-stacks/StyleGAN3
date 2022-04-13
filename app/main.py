from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.routes.root_routes import router as root_routes

app = FastAPI(title="StyleGAN3 API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )

app.include_router(root_routes)
