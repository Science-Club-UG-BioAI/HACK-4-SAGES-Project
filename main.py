from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from Backend.model import load_model, predict, save_prediction_boxplots
from ML.main import target_cols
import torch

import numpy as np


if not Path("best_model.pth").exists():
    hf_hub_download(repo_id="kodON/MultiHeadInaraRegressor", filename="best_model.pt")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model("best_model.pt", DEVICE)


@app.get("/health/")
async def health():
    return {"status": "ok"}


@app.post("/upload/")
async def upload(
    file: UploadFile = File(...),
    num_repeats: int = Form(...),
    plot_path: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="File type need to be .csv or txt")

    if not plot_path.endswith((".png", ".jpg")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file path for plots: file needs to be .jpg or .png",
        )

    Path(plot_path).parent.mkdir(exist_ok=True, parents=True)

    try:
        arr = np.fromstring(
            (await file.read()).decode("utf-8"), sep=",", dtype=np.float32
        )
        pred, mean, std, err = predict(model, arr, num_repeats)
        save_prediction_boxplots(pred, mean, err, target_cols, plot_path, 4)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    return {
        "predicitons": pred.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "errors": err.tolist(),
        "plot_path": plot_path,
    }
