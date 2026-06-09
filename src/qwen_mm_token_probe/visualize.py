from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .probe import TokenScore


def write_generated_text(path: str | Path, text: str) -> Path:
    out_path = Path(path)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_scores_csv(path: str | Path, scores: Iterable["TokenScore"]) -> Path:
    out_path = Path(path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "token_id",
                "token",
                "p_original",
                "p_masked",
                "delta_p",
                "logp_original",
                "logp_masked",
                "delta_logp",
            ],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(score.to_dict())
    return out_path


def write_scores_json(path: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(path)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def write_probability_plot(path: str | Path, scores: list["TokenScore"]) -> Path:
    out_path = Path(path)
    if not scores:
        raise ValueError("cannot plot empty token scores")

    x = [score.index for score in scores]
    p_original = [score.p_original for score in scores]
    p_masked = [score.p_masked for score in scores]

    width = max(10.0, min(28.0, len(scores) * 0.28))
    fig, ax = plt.subplots(figsize=(width, 5.5), constrained_layout=True)
    ax.plot(x, p_original, marker="o", markersize=3, linewidth=1.6, label="original image")
    ax.plot(x, p_masked, marker="o", markersize=3, linewidth=1.6, label="masked image")
    ax.fill_between(x, p_original, p_masked, alpha=0.12)

    tick_stride = max(1, len(scores) // 40)
    ax.set_xticks(x[::tick_stride])
    ax.set_xticklabels([scores[i].compact_token for i in range(0, len(scores), tick_stride)], rotation=70, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("generated token")
    ax.set_ylabel("teacher-forced probability")
    ax.set_title("Per-token probability under original vs randomly masked image")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_html_report(
    path: str | Path,
    *,
    model_id: str,
    prompt: str,
    generated_text: str,
    scores: list["TokenScore"],
    metadata: dict[str, object],
) -> Path:
    out_path = Path(path)
    token_rows = "\n".join(_table_row(score) for score in scores)
    original_strip = "".join(_token_span(score.token, score.p_original) for score in scores)
    masked_strip = "".join(_token_span(score.token, score.p_masked) for score in scores)
    metadata_json = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Token Probability Probe</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    body {{
      margin: 0;
      background: #f7f7f4;
      color: #1f2528;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
    }}
    h2 {{
      font-size: 19px;
      margin-top: 28px;
    }}
    .meta {{
      color: #506168;
      margin-bottom: 24px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #dfe3df;
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
    }}
    .strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 3px;
      margin-top: 8px;
    }}
    .tok {{
      border: 1px solid rgba(31, 37, 40, 0.12);
      border-radius: 5px;
      padding: 2px 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #dfe3df;
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #edf0ed;
      padding: 8px 10px;
      text-align: right;
      vertical-align: top;
    }}
    th {{
      background: #edf2ef;
      color: #344145;
      font-weight: 650;
    }}
    td.token, th.token {{
      text-align: left;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
<main>
  <h1>Token Probability Probe</h1>
  <div class="meta">Model: {html.escape(model_id)}</div>

  <section class="panel">
    <h2>Prompt</h2>
    <pre>{html.escape(prompt)}</pre>
  </section>

  <section class="panel">
    <h2>Generated Text</h2>
    <pre>{html.escape(generated_text)}</pre>
  </section>

  <section class="panel">
    <h2>Original Image Token Probabilities</h2>
    <div class="strip">{original_strip}</div>
  </section>

  <section class="panel">
    <h2>Masked Image Token Probabilities</h2>
    <div class="strip">{masked_strip}</div>
  </section>

  <h2>Scores</h2>
  <table>
    <thead>
      <tr>
        <th>idx</th>
        <th>token id</th>
        <th class="token">token</th>
        <th>p original</th>
        <th>p masked</th>
        <th>delta p</th>
        <th>logp original</th>
        <th>logp masked</th>
        <th>delta logp</th>
      </tr>
    </thead>
    <tbody>
      {token_rows}
    </tbody>
  </table>

  <section class="panel">
    <h2>Metadata</h2>
    <pre>{metadata_json}</pre>
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")
    return out_path


def _table_row(score: "TokenScore") -> str:
    return f"""<tr>
  <td>{score.index}</td>
  <td>{score.token_id}</td>
  <td class="token">{html.escape(score.token)}</td>
  <td>{score.p_original:.6f}</td>
  <td>{score.p_masked:.6f}</td>
  <td>{score.delta_p:+.6f}</td>
  <td>{score.logp_original:.6f}</td>
  <td>{score.logp_masked:.6f}</td>
  <td>{score.delta_logp:+.6f}</td>
</tr>"""


def _token_span(token: str, probability: float) -> str:
    safe_probability = min(1.0, max(0.0, probability))
    hue = 8 + 128 * safe_probability
    background = f"hsl({hue:.1f} 70% 88%)"
    title = f"p={probability:.6f}"
    return (
        f'<span class="tok" title="{html.escape(title)}" '
        f'style="background: {background};">{html.escape(token)}</span>'
    )
