#!/usr/bin/env python3
"""Check and smoke-test the environment used by the arXiv-to-VERL pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


ENGINE_TOOLS = {
    "pdflatex": ("pdflatex",),
    "xelatex": ("xelatex",),
    "latex_dvips_ps2pdf": ("latex", "dvips", "ps2pdf"),
}
ENGINE_FLAGS = {
    "pdflatex": "-pdf",
    "xelatex": "-xelatex",
    "latex_dvips_ps2pdf": "-pdfps",
}
PYTHON_MODULES = ("pdfplumber", "PIL", "pyarrow")
TEX_RESOURCES = ("article.cls", "amsmath.sty", "hyperref.sty")


def executable(value: str | Path) -> Path | None:
    path = Path(str(value)).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path.absolute()
    found = shutil.which(str(value))
    return Path(found).absolute() if found else None


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def tex_resource(kpsewhich: Path | None, name: str) -> dict[str, Any]:
    if kpsewhich is None:
        return {"status": "missing", "path": None, "reason": "kpsewhich_unavailable"}
    result = run([str(kpsewhich), name], timeout=30)
    path = result["stdout_tail"].strip().splitlines()
    resolved = path[-1] if path else ""
    return {
        "status": "passed" if result["return_code"] == 0 and resolved else "missing",
        "path": resolved or None,
    }


def compile_engine(
    *,
    engine: str,
    latexmk: Path,
    pdftoppm: Path,
    root: Path,
) -> dict[str, Any]:
    missing = [name for name in ENGINE_TOOLS[engine] if executable(name) is None]
    if missing:
        return {"status": "unavailable", "missing_tools": missing}
    build_dir = root / engine
    build_dir.mkdir(parents=True)
    source = build_dir / "smoke.tex"
    source.write_text(
        """\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{hyperref}
\\begin{document}
Environment smoke test. Inline math $x^2+y^2=z^2$.
\\section{Heading}
Visible text for PDF and PNG generation.
\\end{document}
""",
        encoding="utf-8",
    )
    command = [
        str(latexmk),
        "-norc",
        "-g",
        ENGINE_FLAGS[engine],
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-no-shell-escape",
        "smoke.tex",
    ]
    compile_result = run(command, cwd=build_dir, timeout=180)
    pdf = build_dir / "smoke.pdf"
    if compile_result["return_code"] != 0 or not pdf.is_file() or pdf.stat().st_size == 0:
        return {
            "status": "failed",
            "compile": compile_result,
            "pdf": str(pdf),
        }
    render = run(
        [
            str(pdftoppm),
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-png",
            "-r",
            "96",
            str(pdf),
            str(build_dir / "smoke_page"),
        ],
        timeout=60,
    )
    png = build_dir / "smoke_page.png"
    status = (
        "passed"
        if render["return_code"] == 0 and png.is_file() and png.stat().st_size > 0
        else "failed"
    )
    return {
        "status": status,
        "compile": compile_result,
        "render": render,
        "pdf_bytes": pdf.stat().st_size,
        "png_bytes": png.stat().st_size if png.is_file() else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latexmk", default=shutil.which("latexmk") or "latexmk")
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm") or "pdftoppm")
    parser.add_argument(
        "--require-all-engines",
        action="store_true",
        help="fail unless pdflatex, xelatex and latex->dvips->ps2pdf all work",
    )
    parser.add_argument("--json-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    print(f"[start] python={sys.executable} version={sys.version.split()[0]}", flush=True)
    python_results: dict[str, Any] = {}
    for module_name in PYTHON_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            python_results[module_name] = {"status": "passed", "version": str(version)}
        except Exception as exc:  # noqa: BLE001
            python_results[module_name] = {
                "status": "missing",
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(
            f"[check] kind=python name={module_name} "
            f"status={python_results[module_name]['status']}",
            flush=True,
        )

    latexmk = executable(args.latexmk)
    pdftoppm = executable(args.pdftoppm)
    kpsewhich = executable("kpsewhich")
    tools = {
        name: str(path) if path else None
        for name, path in {
            "latexmk": latexmk,
            "pdflatex": executable("pdflatex"),
            "xelatex": executable("xelatex"),
            "latex": executable("latex"),
            "dvips": executable("dvips"),
            "ps2pdf": executable("ps2pdf"),
            "pdftoppm": pdftoppm,
            "kpsewhich": kpsewhich,
        }.items()
    }
    for name, path in tools.items():
        print(
            f"[check] kind=executable name={name} status={'passed' if path else 'missing'} "
            f"path={path or '-'}",
            flush=True,
        )
    resources = {name: tex_resource(kpsewhich, name) for name in TEX_RESOURCES}
    for name, value in resources.items():
        print(
            f"[check] kind=tex_resource name={name} status={value['status']} "
            f"path={value.get('path') or '-'}",
            flush=True,
        )

    engines: dict[str, Any] = {}
    if latexmk is not None and pdftoppm is not None:
        with tempfile.TemporaryDirectory(prefix="arxiv_verl_env_") as directory:
            root = Path(directory)
            for index, engine in enumerate(ENGINE_FLAGS, start=1):
                print(
                    f"[compile-start] engine={engine} unit={index}/{len(ENGINE_FLAGS)}",
                    flush=True,
                )
                engines[engine] = compile_engine(
                    engine=engine,
                    latexmk=latexmk,
                    pdftoppm=pdftoppm,
                    root=root,
                )
                print(
                    f"[compile-done] engine={engine} status={engines[engine]['status']} "
                    f"unit={index}/{len(ENGINE_FLAGS)}",
                    flush=True,
                )
    else:
        engines = {
            engine: {"status": "unavailable", "missing_tools": ["latexmk_or_pdftoppm"]}
            for engine in ENGINE_FLAGS
        }

    python_ok = all(row["status"] == "passed" for row in python_results.values())
    resources_ok = all(row["status"] == "passed" for row in resources.values())
    passed_engines = [name for name, row in engines.items() if row["status"] == "passed"]
    engine_ok = (
        len(passed_engines) == len(engines)
        if args.require_all_engines
        else bool(passed_engines)
    )
    status = "passed" if python_ok and resources_ok and engine_ok else "failed"
    report = {
        "status": status,
        "require_all_engines": args.require_all_engines,
        "python": python_results,
        "executables": tools,
        "tex_resources": resources,
        "engines": engines,
        "passed_engines": passed_engines,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"[finish] status={status} passed_engines={','.join(passed_engines) or '-'} "
        f"python_ok={python_ok} tex_resources_ok={resources_ok} "
        f"elapsed={report['elapsed_seconds']}s",
        flush=True,
    )
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
