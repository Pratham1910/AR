"""FastAPI application entrypoint (Project.md #10, #63)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import assets, evidence, inspection, procedures, vision

app = FastAPI(
    title="TVASTA Industrial AR/MR Procedure & QA Platform",
    version="0.1.0",
    description="Phase 1 — camera-based procedure execution and deterministic QA.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assets.router)
app.include_router(procedures.router)
app.include_router(inspection.router)
app.include_router(vision.router)
app.include_router(evidence.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
