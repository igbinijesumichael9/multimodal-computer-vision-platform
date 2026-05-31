from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Multimodal Computer Vision API",
    version="1.0.0"
)

# Include API routes
app.include_router(router)

# Serve output images
app.mount(
    "/outputs",
    StaticFiles(directory="data/outputs"),
    name="outputs"
)