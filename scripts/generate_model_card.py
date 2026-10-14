"""Generate docs/MODEL_CARD.md from REAL artefacts (model_meta.json + metrics.json). No invented numbers.
  python scripts/generate_model_card.py mobilenetv3s-v1
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "ml" / "models" / sys.argv[1]
meta = json.loads((run / "model_meta.json").read_text())
m = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else None
rw = run / "eval_realworld" / "metrics.json"
rw = json.loads(rw.read_text()) if rw.exists() else None

rows = ""
if m:
    rows = "\n".join(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |"
                     for k, v in m["per_class"].items() if isinstance(v, dict) and "precision" in v)
if m:
    metrics_line = ("**Accuracy:** %.4f | **Macro precision:** %.4f | **Macro recall:** %.4f | **Macro F1:** %.4f"
                    % (m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"]))
    n_img = f", {m['num_images']} images"
else:
    metrics_line, n_img = "_Run evaluate.py first._", ""
if rw:
    rw_line = "Accuracy %.4f, macro F1 %.4f on %d images (see eval_realworld/)." % (rw["accuracy"], rw["macro_f1"], rw["num_images"])
else:
    rw_line = "**Not performed yet.** Do not claim field robustness until this is filled in."
unc = meta.get("uncertainty")
unc_line = {k: unc[k] for k in ("confidence", "margin", "entropy")} if unc else "defaults (not calibrated)"
crops = sorted({c.split("___")[0] for c in meta["class_names"]})
text = f"""# Model Card - KrishiDrishti {meta['model_version']}

*Auto-generated from `ml/models/{sys.argv[1]}`. Do not hand-edit numbers.*

## Model
- Architecture: **{meta['architecture']}** (transfer learning, ImageNet weights; {meta['params']:,} parameters)
- Input: RGB leaf image resized to {meta['input_size']}x{meta['input_size']}; {meta['input_range']}
- Trained: {meta['trained_at']}  (training time {meta['training_seconds']/60:.1f} min)
- Dataset: {meta['dataset']['name']} ({meta['dataset']['version']}), split counts {meta['dataset']['split_counts']}

## Intended use
AI-assisted visual screening of **single crop leaves** to support (not replace) agricultural expert advice.

## Supported crops: {', '.join(crops)}
## Supported classes ({len(meta['class_names'])})
{chr(10).join('- ' + c for c in meta['class_names'])}

## Metrics (held-out test split{n_img})
{metrics_line}

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{rows}

## Real-world (field images) evaluation
{rw_line}

## Uncertainty handling
Thresholds: {unc_line}.
Low-reliability outputs are reported as "Unable to make a reliable prediction".

## Known limitations
- Trained on leaf images only; not a general plant diagnosis system.
- Controlled-background datasets (e.g. PlantVillage) overstate field performance.
- Only the listed crops/diseases are supported; anything else may be mislabelled as the closest class.
- Softmax confidence is not a perfect out-of-distribution detector.
- Severity is a colour-threshold heuristic (see docs/SEVERITY_METHOD.md); unvalidated unless a validation report exists.
- Grad-CAM shows model attention, not biological causation.

## Out-of-scope inputs
Whole-field photos, fruits/stems/roots, non-listed crops, blurred/dark images, multiple overlapping leaves.
"""
(ROOT / "docs" / "MODEL_CARD.md").write_text(text, encoding="utf-8")
print("wrote docs/MODEL_CARD.md")
