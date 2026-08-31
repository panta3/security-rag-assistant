from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router

app = FastAPI(title="Security RAG Assistant")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    # Single self-contained HTML file (inline CSS/JS) — no separate
    # static-assets mount needed since there's nothing else to serve yet.
    return FileResponse("static/index.html")
