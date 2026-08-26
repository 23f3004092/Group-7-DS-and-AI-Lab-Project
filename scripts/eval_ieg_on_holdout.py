"""
eval_ieg_on_holdout.py
======================
Evaluates the trained IEG checkpoint on kcc_eval_1_augmented.csv.

Two evaluation inputs per row:
  1. QueryText  — raw KCC query column
  2. first_user_prompt — first user turn extracted from multi_turn_json

Key considerations:
  - Eval CSV has 9 intent classes (100 each = 900 rows), perfectly balanced.
  - Model was trained on 10 classes (adds non_agri). non_agri is NOT in eval set.
  - Guardrail label: none of the eval rows are expected to be blocked (all real KCC).
   - Metrics reported per-class and macro-averaged; also per-language breakdown.
   - Weighted F1 used for the "overall" headline number (comparable across distributions).

Confusion / soft-routing analysis (for offline confusion-prior routing expansion):
   - Raw + row-normalized confusion matrices per checkpoint × input column.
   - Ranked confusion pairs (true → predicted) above a rate floor.
   - Soft-routing survival stats mirroring production ieg_run() in run_e2e_eval.py:
     does the TRUE class survive the top-3 + softmax>0.15 routing set?
   - Top-1 confidence calibration buckets (accuracy per confidence decile) to pick
     a confidence-gate threshold for loosening retrieval filters.
   - Ready-to-merge INTENT_QTYPE_MAP expansion snippet (intent -> confused intents).
"""

import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
EVAL_CSV    = ROOT / "data" / "processed" / "kcc" / "kcc_eval_1_augmented.csv"
RESULTS_OUT = ROOT / "data" / "eval" / "ieg_holdout_results.json"
RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Checkpoint registry ───────────────────────────────────────────────────────
# "deployed"    = the model currently live in run_e2e_eval.py (aneeqasiddiqui377/v4-output)
# "experimental"= local outputs/ieg_model (new 10-class training)
CHECKPOINTS = {
    "deployed": {
        "source":      "kagglehub",
        "dataset_id":  "aneeqasiddiqui377/v4-output",
        "ckpt_name":   "ieg_adamw",   # prefix to match
        "description": "Deployed v4 (hing-mbert, 7 classes, M5 fine-tune)",
    },
    "experimental": {
        "source":      "local",
        "ieg_out_dir": ROOT / "outputs" / "ieg_final_latest" / "outputs" / "ieg_model",
        "ckpt_name":   "ieg_adamw",
        "description": "Experimental (hing-mbert, 10 classes, new guardrail aug)",
    },
}

MAX_LEN = 64   # safe upper bound; tokenizer always truncates
DEVICE  = "cpu"

# ── Confusion-matrix / soft-routing analysis config ───────────────────────────
# These mirror production routing in run_e2e_eval.py::ieg_run()
SOFTMAX_ROUTING_THRESH = 0.15   # any intent with softmax prob > this joins the routing set
TOP_K_ROUTING          = 3      # top-k intents always included in routing set
CONFUSION_MIN_RATE     = 0.05   # min row-normalized confusion rate to recommend a pair
CONFUSION_MIN_COUNT    = 5      # min absolute count to recommend a pair (noise floor)
CONFUSION_OUT = ROOT / "data" / "eval" / "ieg_confusion_matrix.json"

def detect_model_name(ckpt_path: Path) -> str:
    """
    Infer HuggingFace model name from the checkpoint's backbone key pattern.
      DistilBERT: backbone.transformer.layer.*   (6 layers, q_lin/k_lin)
      mBERT/BERT: backbone.encoder.layer.*        (12 layers, query/key)
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    keys = list(sd.keys())

    # DistilBERT uses 'transformer.layer' (not 'encoder.layer')
    if any(k.startswith("backbone.transformer.layer.") for k in keys):
        name = "distilbert-base-multilingual-cased"
        n_layers = max(int(k.split(".")[3]) for k in keys if k.startswith("backbone.transformer.layer.")) + 1
        print(f"  Detected backbone: DistilBERT ({n_layers} transformer layers) -> {name}")
        return name

    # BERT-family uses 'encoder.layer'
    bert_layers = [int(k.split(".")[3]) for k in keys if k.startswith("backbone.encoder.layer.")]
    if bert_layers:
        n_layers = max(bert_layers) + 1
        name = "l3cube-pune/hing-mbert-mixed" if n_layers == 12 else "bert-base-multilingual-cased"
        print(f"  Detected backbone: BERT ({n_layers} encoder layers) -> {name}")
        return name

    print("  WARNING: Could not detect backbone; defaulting to bert-base-multilingual-cased")
    return "bert-base-multilingual-cased"


# ── IEG model architecture (copied verbatim from run_e2e_eval.py) ─────────────
class IEGModel(nn.Module):
    def __init__(self, n_intents: int, n_ner: int, model_name: str):
        super().__init__()
        self.backbone       = AutoModel.from_pretrained(model_name)
        h                   = self.backbone.config.hidden_size
        self.intent_head    = nn.Linear(h, n_intents)
        self.ner_head       = nn.Linear(h, n_ner)
        self.guardrail_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0, :]
        return self.intent_head(pooled), self.ner_head(out), self.guardrail_head(pooled)


# ── Resolve a checkpoint from the registry entry ────────────────────────────────
def resolve_ckpt_and_labels(cfg: dict) -> tuple:
    """Returns (ckpt_path, label_maps_path) for a given CHECKPOINTS entry."""
    if cfg["source"] == "kagglehub":
        import kagglehub
        base = Path(kagglehub.dataset_download(cfg["dataset_id"]))
        pts  = list(base.rglob("*.pt"))
        lms  = list(base.rglob("label_maps.json"))
    else:  # local
        base = cfg["ieg_out_dir"]
        pts  = list(base.glob("*.pt"))
        lms  = list(base.glob("label_maps.json"))

    prefix = cfg["ckpt_name"]
    ckpt = next((p for p in pts if prefix in p.name), None)
    if ckpt is None and pts:
        ckpt = pts[0]

    label_map_path = lms[0] if lms else None
    return ckpt, label_map_path


# ── Load a checkpoint ─────────────────────────────────────────────────────
def load_ieg_from_cfg(cfg: dict):
    """Load IEG model from a CHECKPOINTS config entry."""
    ckpt, label_map_path = resolve_ckpt_and_labels(cfg)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for {cfg['description']}")

    print(f"  Checkpoint : {ckpt.name}")

    if label_map_path and label_map_path.exists():
        with open(label_map_path) as f:
            label_maps = json.load(f)
    else:
        label_maps = {}

    intent_classes = label_maps.get("intent_classes", [
        "cultivation_practice", "disease_pest", "general", "non_agri",
        "nutrition_fertilizer", "post_harvest_storage", "specialty_other"
    ])
    ner_labels = label_maps.get("ner_labels", ["O", "B-CROP", "I-CROP", "B-DISTRICT", "I-DISTRICT"])

    model_name = detect_model_name(ckpt)
    print(f"  Backbone   : {model_name}  ({len(intent_classes)} intent classes)")
    print(f"  Classes    : {intent_classes}")

    model = IEGModel(n_intents=len(intent_classes), n_ner=len(ner_labels), model_name=model_name)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    tok = AutoTokenizer.from_pretrained(model_name)
    id2intent = {i: c for i, c in enumerate(intent_classes)}
    intent2id = {c: i for i, c in id2intent.items()}
    return model, tok, id2intent, intent2id, model_name


# ── Single inference ─────────────────────────────────────────────────────────
def predict_intent(model, tok, id2intent, text: str, threshold: float = 0.15):
    """Returns top predicted intent (argmax), all intents above softmax threshold, and guardrail flag.
    Uses SOFTMAX + 0.15 to mirror production routing (CrossEntropyLoss training)."""
    enc = tok(
        str(text), truncation=True, max_length=MAX_LEN,
        padding="max_length", return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        il, nl, gl = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    probs_sm   = torch.softmax(il[0], dim=-1).cpu().numpy()
    top_intent = id2intent[int(np.argmax(probs_sm))]
    multi_intents = [id2intent[i] for i, p in enumerate(probs_sm) if p > threshold]
    guardrail  = int(gl.argmax(-1).item()) == 1

    return top_intent, multi_intents, guardrail, probs_sm, probs_sm


# ── Extract first user prompt from multi_turn_json ───────────────────────────
def get_first_user_prompt(cell) -> str:
    try:
        turns = json.loads(cell) if isinstance(cell, str) else cell
        for turn in turns:
            if turn.get("role") == "user":
                return turn.get("content", "")
    except Exception:
        pass
    return ""


# ── Confusion-matrix + soft-routing analysis ─────────────────────────────────
def analyze_confusion_and_routing(true_labels, preds, probs_mx, id2intent):
    """
    Builds everything needed for offline confusion-prior routing expansion:
      1. Raw + row-normalized confusion matrices.
      2. Ranked confusion pairs (true -> predicted).
      3. Soft-routing survival stats mirroring ieg_run() in run_e2e_eval.py:
         does the TRUE class survive the softmax top-3 + >0.15 routing set?
      4. Top-1 confidence calibration buckets (for confidence-gated filter loosening).
      5. Ready-to-merge expansion recommendations (intent -> confusions).

    probs_mx: (N, n_model_classes) SOFTMAX probabilities.
    """
    model_classes = list(id2intent.values())
    classes       = sorted(set(true_labels) | set(preds))
    correct       = np.array([t == p for t, p in zip(true_labels, preds)])

    # ── 1. Confusion matrices ────────────────────────────────────────────────
    cm = confusion_matrix(true_labels, preds, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums,
                        out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    # ── 2. Ranked confusion pairs ────────────────────────────────────────────
    pairs = []
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": t, "pred": p,
                    "count": int(cm[i, j]),
                    "rate": round(float(cm_norm[i, j]), 4),
                })
    pairs.sort(key=lambda d: (-d["count"], d["true"], d["pred"]))
    top_pairs = [p for p in pairs if p["rate"] >= CONFUSION_MIN_RATE][:25]

    # ── 3. Soft-routing survival stats (mirrors production ieg_run) ──────────
    n_cls = probs_mx.shape[1]
    k     = min(TOP_K_ROUTING, n_cls)
    order     = np.argsort(-probs_mx, axis=1)
    top1_conf = probs_mx.max(axis=1)
    entropy   = -np.sum(np.where(probs_mx > 0, probs_mx * np.log(probs_mx + 1e-12), 0.0), axis=1)

    intent2id_local = {c: i for i, c in enumerate(model_classes)}
    true_ids = np.array([intent2id_local.get(t, -1) for t in true_labels])
    has_true = true_ids >= 0   # rows whose true class exists in the model's output space

    rank_of_true = np.full(len(true_labels), -1)
    for r in np.where(has_true)[0]:
        pos = np.where(order[r] == true_ids[r])[0]
        if len(pos):
            rank_of_true[r] = int(pos[0])

    in_topk   = np.where(has_true, rank_of_true < k, False)
    in_thresh = np.array([
        bool(has_true[r] and probs_mx[r, true_ids[r]] > SOFTMAX_ROUTING_THRESH)
        for r in range(len(true_labels))
    ])

    def _bucket(mask):
        sel  = mask[has_true]
        base = int(has_true.sum())
        return {
            "count": int(sel.sum()),
            "pct":   round(float(sel.mean()) * 100, 2) if base else 0.0,
        }

    routing = {
        "softmax": True,
        "top_k": k,
        "threshold": SOFTMAX_ROUTING_THRESH,
        "rows_with_true_in_model": int(has_true.sum()),
        "argmax_hit":          _bucket(correct),
        "true_in_topk":        _bucket(in_topk),
        "true_in_threshold":   _bucket(in_thresh),
        "true_in_union":       _bucket(in_topk | in_thresh),
        "mean_top1_conf_correct": round(float(top1_conf[correct].mean()), 4) if correct.any() else None,
        "mean_top1_conf_wrong":   round(float(top1_conf[~correct].mean()), 4) if (~correct).any() else None,
        "mean_entropy_correct":   round(float(entropy[correct].mean()), 4) if correct.any() else None,
        "mean_entropy_wrong":     round(float(entropy[~correct].mean()), 4) if (~correct).any() else None,
    }

    # ── 4. Confidence calibration (accuracy per top-1 confidence decile) ─────
    calibration = []
    bins = np.linspace(0.0, 1.0001, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (top1_conf >= lo) & (top1_conf < hi)
        if m.sum():
            calibration.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "n":     int(m.sum()),
                "acc":   round(float(correct[m].mean()), 4),
            })

    # ── 5. Expansion recommendations (ready-to-merge into INTENT_QTYPE_MAP) ──
    recs = {}
    for p_ in pairs:
        if (p_["rate"] >= CONFUSION_MIN_RATE
                and p_["count"] >= CONFUSION_MIN_COUNT
                and p_["pred"] in model_classes):
            recs.setdefault(p_["true"], []).append(
                {"intent": p_["pred"], "rate": p_["rate"], "count": p_["count"]})
    rec_snippet = {t: sorted(r["intent"] for r in v) for t, v in sorted(recs.items())}

    return {
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_norm": [[round(float(x), 4) for x in row] for row in cm_norm],
        "confusion_pairs_all": pairs[:50],
        "confusion_pairs_reported": top_pairs,
        "routing_stats": routing,
        "confidence_calibration": calibration,
        "expansion_recommendations": recs,
        "expansion_snippet": rec_snippet,
    }


def print_analysis(res: dict, source_name: str):
    """Human-readable dump of the confusion/routing analysis."""
    classes = res["classes"]
    cmn     = res["confusion_matrix_row_norm"]

    print(f"\n  --- Confusion matrix (row-normalized, rows=true cols=pred) [{source_name}] ---")
    short = {c: c[:10] for c in classes}
    hdr = "true \\ pred".ljust(14) + "".join(short[c].ljust(11) for c in classes)
    print(hdr)
    for i, t in enumerate(classes):
        row = short[t].ljust(14) + "".join(f"{cmn[i][j]:<11.3f}" for j in range(len(classes)))
        print(row)

    if res["confusion_pairs_reported"]:
        print(f"\n  Top confusion pairs (rate >= {CONFUSION_MIN_RATE:.2f}):")
        for p_ in res["confusion_pairs_reported"][:15]:
            print(f"    {p_['true']:<28} -> {p_['pred']:<28} "
                  f"count={p_['count']:>3}  rate={p_['rate']*100:5.1f}%")

    r = res["routing_stats"]
    print(f"\n  Soft-routing survival (softmax top-{r['top_k']} + prob>{r['threshold']}):")
    print(f"    argmax hit              : {r['argmax_hit']['pct']:5.1f}%")
    print(f"    true in top-{r['top_k']}            : {r['true_in_topk']['pct']:5.1f}%")
    print(f"    true in >{r['threshold']} set      : {r['true_in_threshold']['pct']:5.1f}%")
    print(f"    true in UNION (routed)  : {r['true_in_union']['pct']:5.1f}%")
    if r["mean_top1_conf_correct"] is not None:
        print(f"    mean top-1 conf  correct/wrong : {r['mean_top1_conf_correct']:.3f} / {r['mean_top1_conf_wrong']:.3f}")
        print(f"    mean entropy     correct/wrong : {r['mean_entropy_correct']:.3f} / {r['mean_entropy_wrong']:.3f}")

    if res["confidence_calibration"]:
        print("\n  Confidence calibration (accuracy per top-1 conf bucket):")
        for b in res["confidence_calibration"]:
            bar = "#" * int(b["acc"] * 40)
            print(f"    {b['range']:<9} n={b['n']:>4}  acc={b['acc']:.3f}  {bar}")

    if res["expansion_snippet"]:
        print(f"\n  Suggested INTENT_QTYPE_MAP expansion (rate>={CONFUSION_MIN_RATE:.2f}, count>={CONFUSION_MIN_COUNT}):")
        print("    EXPANSION_PRIORS = " + json.dumps(res["expansion_snippet"], indent=8))


# ── Evaluate one batch of texts ───────────────────────────────────────────────
def evaluate_texts(model, tok, id2intent, intent2id, texts, true_labels, source_name: str):
    """
    Evaluate model on a list of texts against true_labels.
    Returns per-class and macro metrics.
    
    IMPORTANT: eval CSV has classes not in model (and vice versa).
    We evaluate only on classes that appear in BOTH sets.
    """
    preds = []
    guardrails = []
    sm_probs = []
    for text in texts:
        top, _, gr, _, sm = predict_intent(model, tok, id2intent, text)
        preds.append(top)
        guardrails.append(gr)
        sm_probs.append(sm)

    # Classes present in eval set
    eval_classes   = sorted(set(true_labels))
    model_classes  = set(id2intent.values())

    # Classes in eval but NOT in model → these will always be wrong
    unseen_in_model = set(eval_classes) - model_classes
    if unseen_in_model:
        print(f"\n  ⚠️  Classes in eval NOT in model: {unseen_in_model}")
        print(f"     These will always be misclassified (model can't output them).")

    # Classes in model but NOT in eval → fine, model just won't predict them here
    eval_only = model_classes - set(eval_classes)
    if eval_only:
        print(f"  ℹ️  Classes in model NOT in eval: {eval_only}")

    # Full classification report (all eval classes, including unseen_in_model)
    labels_for_report = sorted(set(true_labels) | set(preds))
    report = classification_report(
        true_labels, preds,
        labels=labels_for_report,
        output_dict=True,
        zero_division=0
    )

    # Headline metrics
    acc          = accuracy_score(true_labels, preds)
    macro_f1     = f1_score(true_labels, preds, average="macro",    labels=eval_classes, zero_division=0)
    weighted_f1  = f1_score(true_labels, preds, average="weighted", labels=eval_classes, zero_division=0)
    guardrail_fp = sum(guardrails)  # all eval rows should be safe → FP count

    print(f"\n{'='*60}")
    print(f"  Source: {source_name}  (n={len(texts)})")
    print(f"{'='*60}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}  (across {len(eval_classes)} eval classes)")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Guardrail FP    : {guardrail_fp} / {len(texts)} ({guardrail_fp/len(texts)*100:.1f}%)")
    print()
    print("  Per-class F1 (eval classes only):")
    for cls in sorted(eval_classes):
        cls_metrics = report.get(cls, {})
        marker = " ← NOT IN MODEL" if cls in unseen_in_model else ""
        print(f"    {cls:<30} P={cls_metrics.get('precision',0):.3f}  R={cls_metrics.get('recall',0):.3f}  F1={cls_metrics.get('f1-score',0):.3f}  n={cls_metrics.get('support',0)}{marker}")

    # ── Confusion / soft-routing analysis (softmax, mirrors production) ──
    analysis = analyze_confusion_and_routing(true_labels, preds, np.vstack(sm_probs), id2intent)
    print_analysis(analysis, source_name)

    return {
        "source": source_name,
        "n": len(texts),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "guardrail_false_positives": guardrail_fp,
        "unseen_in_model": list(unseen_in_model),
        "per_class": {
            cls: {
                "precision": round(report.get(cls, {}).get("precision", 0), 4),
                "recall":    round(report.get(cls, {}).get("recall",    0), 4),
                "f1":        round(report.get(cls, {}).get("f1-score",  0), 4),
                "support":   int(report.get(cls, {}).get("support",     0)),
                "in_model":  cls not in unseen_in_model,
            }
            for cls in sorted(eval_classes)
        },
        "predictions": preds,
        "true_labels": list(true_labels),
        "analysis": analysis,
    }


# ── Per-language breakdown ────────────────────────────────────────────────────
def per_language_metrics(model, tok, id2intent, intent2id, df, text_col, source_name):
    print(f"\n  --- Per-language breakdown ({source_name}) ---")
    results = {}
    for lang, sub in df.groupby("language"):
        texts  = sub[text_col].fillna("").astype(str).tolist()
        labels = sub["intent_label"].tolist()
        preds  = [predict_intent(model, tok, id2intent, t)[0] for t in texts]
        eval_classes = sorted(set(labels))
        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average="macro",    labels=eval_classes, zero_division=0)
        wf1 = f1_score(labels, preds, average="weighted", labels=eval_classes, zero_division=0)
        print(f"    {lang:<42} acc={acc:.3f}  macro_f1={mf1:.3f}  weighted_f1={wf1:.3f}  n={len(texts)}")
        results[lang] = {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4), "n": len(texts)}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("IEG Holdout Evaluation — kcc_eval_1_augmented.csv")
    print("Comparing: DEPLOYED (v4-output) vs EXPERIMENTAL (local)")
    print("=" * 60)

    # Load eval data
    df = pd.read_csv(EVAL_CSV)
    print(f"\nEval set: {len(df)} rows")
    print(f"Intent classes in eval : {sorted(df['intent_label'].unique())}")
    print(f"Rows per class         : {df['intent_label'].value_counts().to_dict()}")

    # Extract first user prompt
    df["first_user_prompt"] = df["multi_turn_json"].apply(get_first_user_prompt)
    df["first_user_prompt"] = df.apply(
        lambda r: r["first_user_prompt"] if r["first_user_prompt"] else r["QueryText"], axis=1
    )

    all_results = {}

    for ckpt_key, cfg in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"  MODEL: {cfg['description']}")
        print(f"  Key  : {ckpt_key}")
        print(f"{'='*60}")
        try:
            model, tok, id2intent, intent2id, detected_model = load_ieg_from_cfg(cfg)
        except Exception as e:
            print(f"  ERROR loading {ckpt_key}: {e}")
            all_results[ckpt_key] = {"error": str(e)}
            continue

        qt_results = evaluate_texts(
            model, tok, id2intent, intent2id,
            texts       = df["QueryText"].fillna("").astype(str).tolist(),
            true_labels = df["intent_label"].tolist(),
            source_name = f"{ckpt_key} / QueryText"
        )
        qt_lang = per_language_metrics(model, tok, id2intent, intent2id, df, "QueryText", f"{ckpt_key}/QueryText")

        fu_results = evaluate_texts(
            model, tok, id2intent, intent2id,
            texts       = df["first_user_prompt"].fillna("").astype(str).tolist(),
            true_labels = df["intent_label"].tolist(),
            source_name = f"{ckpt_key} / first_user_prompt"
        )
        fu_lang = per_language_metrics(model, tok, id2intent, intent2id, df, "first_user_prompt", f"{ckpt_key}/first_user_prompt")

        all_results[ckpt_key] = {
            "description": cfg["description"],
            "backbone":    detected_model,
            "n_model_classes": len(id2intent),
            "model_classes":   list(id2intent.values()),
            "QueryText": {
                "overall":      {k: v for k, v in qt_results.items() if k not in ("predictions", "true_labels")},
                "per_language": qt_lang,
            },
            "first_user_prompt": {
                "overall":      {k: v for k, v in fu_results.items() if k not in ("predictions", "true_labels")},
                "per_language": fu_lang,
            },
        }

    # ── Save ──────────────────────────────────────────────────────────
    summary = {
        "eval_csv":            str(EVAL_CSV),
        "n_eval_rows":         len(df),
        "eval_intent_classes": sorted(df["intent_label"].unique().tolist()),
        "checkpoints":         all_results,
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Dedicated confusion-matrix / routing-priors artifact ──────────
    conf_report = {}
    for key, res in all_results.items():
        if "error" in res:
            continue
        for input_col in ["QueryText", "first_user_prompt"]:
            analysis = res[input_col]["overall"].get("analysis")
            if analysis:
                conf_report[f"{key}/{input_col}"] = analysis

    with open(CONFUSION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "softmax_routing_thresh": SOFTMAX_ROUTING_THRESH,
                "top_k_routing": TOP_K_ROUTING,
                "confusion_min_rate": CONFUSION_MIN_RATE,
                "confusion_min_count": CONFUSION_MIN_COUNT,
            },
            "models": conf_report,
        }, f, indent=2, ensure_ascii=False)

    # ── Final comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<24} {'Input':<22} {'Acc':>8} {'MacroF1':>9} {'WtdF1':>8} {'GrdFP':>7}")
    print("-" * 70)
    for key, res in all_results.items():
        if "error" in res:
            print(f"  {key:<22} ERROR: {res['error'][:40]}")
            continue
        desc = res['description'][:22]
        for input_col in ["QueryText", "first_user_prompt"]:
            ov = res[input_col]["overall"]
            print(f"  {desc:<22} {input_col:<22} "
                  f"{ov['accuracy']:>8.4f} {ov['macro_f1']:>9.4f} "
                  f"{ov['weighted_f1']:>8.4f} {ov['guardrail_false_positives']:>7}")
        print()

    print(f"\nResults saved -> {RESULTS_OUT}")
    print(f"Confusion/routing priors -> {CONFUSION_OUT}")
    print("=" * 70)

    # ── Merged expansion snippet (first_user_prompt = production distribution) ──
    merged = {}
    for key in ["experimental", "deployed"]:
        res = all_results.get(key, {})
        if "error" in res:
            continue
        snippet = res["first_user_prompt"]["overall"].get("analysis", {}).get("expansion_snippet", {})
        for true_intent, confusions in snippet.items():
            merged.setdefault(true_intent, set()).update(confusions)

    if merged:
        print("\nMERGED EXPANSION PRIORS (from first_user_prompt evals — merge into INTENT_QTYPE_MAP):")
        print(json.dumps({t: sorted(c) for t, c in sorted(merged.items())}, indent=2))

    print()
    print("NOTE: 'deployed' model has 7 intent classes (old schema — no market/policy/weather/other).")
    print("      Eval CSV has 9 classes. Deployed model CANNOT predict market/policy/weather/other.")
    print("      These 400/900 rows will always be misclassified by the deployed model.")
    print("      'experimental' model has 10 classes and covers all 9 eval classes.")


if __name__ == "__main__":
    main()
