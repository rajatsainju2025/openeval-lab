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


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard():
    tpl = jinja.get_template("leaderboard.html")
    index_p = Path("runs/index.json")
    runs = []
    if index_p.exists():
        try:
            payload = json.loads(index_p.read_text())
            runs = payload.get("runs", [])
        except Exception:
            runs = []
    return tpl.render(title="Leaderboard", runs=runs)


@app.get("/run/{file}", response_class=HTMLResponse)
def run_detail(file: str, offset: int = 0, limit: int = 50):
    # security: only allow basenames under runs/
    file = Path(file).name
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    p = Path("runs") / file
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    tpl = jinja.get_template("run_detail.html")
    # slice records for pagination without mutating original
    records = list(data.get("records", []))
    total = len(records)
    page = records[offset: offset + limit] if records else []
    return tpl.render(
        title=f"Run {file}",
        file=file,
        data=data,
        records=page,
        pagination={"offset": offset, "limit": limit, "total": total},
    )
