from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).resolve().parent / "templates"

jinja = Environment(
    loader=FileSystemLoader(TEMPLATES),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title="OpenEval Lab Dashboard")


@app.get("/", response_class=HTMLResponse)
def index():
    tpl = jinja.get_template("index.html")
    data = {}
    p = Path("results.json")
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    return tpl.render(title="OpenEval Lab", data=data)
