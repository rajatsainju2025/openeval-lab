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


@app.get("/compare", response_class=HTMLResponse)
def compare(a: str = "", b: str = ""):
    tpl = jinja.get_template("compare.html")
    index_p = Path("runs/index.json")
    runs = []
    if index_p.exists():
        try:
            payload = json.loads(index_p.read_text())
            runs = payload.get("runs", [])
        except Exception:
            runs = []

    def load_run(name: str):
        if not name:
            return None
        p = Path("runs") / Path(name).name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    run_a = load_run(a)
    run_b = load_run(b)
    return tpl.render(title="Compare Runs", runs=runs, a=a, b=b, run_a=run_a, run_b=run_b)


@app.get("/run/{file}", response_class=HTMLResponse)
def run_detail(file: str, offset: int = 0, limit: int = 50):
    # security: only allow basenames under runs/
    file = Path(file).name
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    p = Path("runs") / file
    data = {}
    error_msg = None
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in run file: {e}"
        except Exception as e:
            error_msg = f"Error loading run file: {e}"
    else:
        error_msg = f"Run file '{file}' not found in runs/ directory"
    
    tpl = jinja.get_template("run_detail.html")
    # slice records for pagination without mutating original
    records = list(data.get("records", []))
    total = len(records)
    page = records[offset : offset + limit] if records else []
    return tpl.render(
        title=f"Run {file}",
        file=file,
        data=data,
        records=page,
        error_msg=error_msg,
        pagination={"offset": offset, "limit": limit, "total": total},
    )


@app.get("/bias/{file}", response_class=HTMLResponse)
def bias_analysis(file: str):
    """Display bias analysis results."""
    file = Path(file).name
    p = Path("bias_analysis") / file
    data = {}
    error_msg = None
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in bias analysis file: {e}"
        except Exception as e:
            error_msg = f"Error loading bias analysis file: {e}"
    else:
        error_msg = f"Bias analysis file '{file}' not found in bias_analysis/ directory"
    
    tpl = jinja.get_template("bias_analysis.html")
    return tpl.render(
        title=f"Bias Analysis {file}",
        file=file,
        data=data,
        error_msg=error_msg,
    )
    """Export a run file in various formats."""
    # security: only allow basenames under runs/
    file = Path(file).name
    p = Path("runs") / file
    
    if not p.exists():
        from fastapi import HTTPException
        raise HTTPException(404, f"Run file '{file}' not found")
    
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(500, f"Error loading run file: {e}")
    
    if format.lower() == "csv":
        # Export records as CSV if available
        records = data.get("records", [])
        if not records:
            from fastapi import HTTPException
            raise HTTPException(400, "No records available for CSV export")
        
        import io
        import csv
        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={file.replace('.json', '.csv')}"}
        )
    else:
        # Default: JSON export
        from fastapi.responses import Response
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={file}"}
        )
