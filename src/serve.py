from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import joblib
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
pipe = joblib.load(MODEL_PATH)

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

app = FastAPI(title="Text Classifier UI")

templates_dir = Path(__file__).resolve().parent.parent / "templates"
env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)

static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class Item(BaseModel):
    text: str

@app.post("/predict", response_class=JSONResponse)
def predict_api(item: Item):
    pred_id = int(pipe.predict([item.text])[0])
    proba = None
    if hasattr(pipe[-1], "predict_proba"):
        proba = float(pipe.predict_proba([item.text])[0].max())
    return {"label_id": pred_id, "label_name": LABELS[pred_id], "confidence": proba}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    template = env.get_template("index.html")
    return template.render(result=None, text="")

@app.post("/", response_class=HTMLResponse)
def predict_form(request: Request, text: str = Form(...)):
    pred_id = int(pipe.predict([text])[0])
    proba = None
    if hasattr(pipe[-1], "predict_proba"):
        proba = float(pipe.predict_proba([text])[0].max())
    template = env.get_template("index.html")
    result = {
        "label_id": pred_id,
        "label_name": LABELS[pred_id],
        "confidence": None if proba is None else f"{proba:.3f}",
    }
    return template.render(result=result, text=text)
