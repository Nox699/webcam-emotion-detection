"""Self-contained, offline evaluation reports and an exportable confusion chart."""
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .storage import atomic_json, atomic_text


def write_report(metrics: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.array(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=metrics["labels"],
             yticklabels=metrics["labels"], xlabel="Predicted expression", ylabel="Labeled expression",
             title="Held-out test confusion matrix")
    for row in range(3):
        for col in range(3):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                color="white" if matrix[row, col] > matrix.max()/2 else "#142033", fontsize=14)
    fig.colorbar(image, ax=axis, label="Test samples")
    chart = directory / "confusion_matrix.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    rows = []
    for label in metrics["labels"] + ["macro avg", "weighted avg"]:
        values = metrics["classification_report"][label]
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{values['precision']:.3f}</td>"
                    f"<td>{values['recall']:.3f}</td><td>{values['f1-score']:.3f}</td>"
                    f"<td>{values['support']:.0f}</td></tr>")
    simulated = metrics["source"] == "simulated"
    note = ("These scores measure generated feature data only. They do not measure webcam accuracy. "
            "Train and test on independently recorded labeled sessions for a meaningful local evaluation."
            if simulated else
            "These scores measure held-out recording sessions for this dataset. They do not establish "
            "accuracy for unseen people or different cameras. Neighboring frames remain correlated "
            "within a session. Record new sessions to check generalization.")
    encoded = base64.b64encode(chart.read_bytes()).decode("ascii")
    sessions = (f"<p><b>Training sessions:</b> {html.escape(', '.join(metrics['train_sessions']))}<br>"
                f"<b>Test sessions:</b> {html.escape(', '.join(metrics['test_sessions']))}</p>"
                if not simulated else "")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expression Lab | Evaluation</title><style>
body{{background:#eef2f8;color:#182437;font:16px/1.65 system-ui,sans-serif;margin:0}}
main{{max-width:940px;margin:40px auto;background:white;padding:44px;border-radius:20px}}
h1{{font-size:36px;line-height:1.2}}.eyebrow{{color:#35767d;text-transform:uppercase;letter-spacing:2px}}
.stats{{display:flex;gap:32px;flex-wrap:wrap}}.stats strong{{display:block;font-size:36px;color:#21646b}}
.note{{background:#fff3d4;padding:20px;border-radius:12px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #d8e0ea}}img{{width:100%;max-width:650px}}
@media(max-width:650px){{main{{margin:0;padding:22px;border-radius:0}}}}
</style><main><div class="eyebrow">Supervised learning / Logistic regression</div>
<h1>{'Simulated assignment' if simulated else 'Personal webcam'} evaluation</h1>
<div class="stats"><div><strong>{metrics['accuracy']:.1%}</strong>test accuracy</div>
<div><strong>{metrics['train_samples']}</strong>training samples</div>
<div><strong>{metrics['test_samples']}</strong>test samples</div></div>
<p class="note">{note}</p><p><b>Split:</b> {html.escape(metrics['split'])}. <b>Seed:</b> {metrics['seed']}.
The scaler is fit only on the training subset. The deployed model is the evaluated model.</p>{sessions}
<h2>Precision, recall, and F1</h2><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>
<th>F1</th><th>Support</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Classification errors</h2><img alt="Confusion matrix" src="data:image/png;base64,{encoded}">
<h2>Interpretation</h2><p>Happy, Sad, and Neutral are facial-expression labels. A face alone does not
provide a reliable measurement of a person's internal emotional state. Eye openness and smile
intensity are rough Haar-detector proxies. The personal model also uses 900 HOG image features.</p>
<p>Lighting, face pose, glasses, camera quality, small training sets, and ambiguous expressions
can change predictions. A high softmax score is a model score, not a calibrated probability of
someone's emotion. The application suppresses low-score and poor-quality predictions.</p>
<h2>Future improvements</h2><p>Use a larger consented dataset with person-disjoint test sets, facial
landmarks, stronger alignment, probability calibration, and a validated CNN. Evaluate additional
expressions separately before adding new labels.</p></main></html>"""
    atomic_json(directory / "metrics.json", metrics)
    destination = directory / "evaluation.html"
    atomic_text(destination, page)
    return destination
