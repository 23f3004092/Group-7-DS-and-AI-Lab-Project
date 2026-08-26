#!/usr/bin/env python3
"""stage_models_gcp.py
=====================
Stage the two model artifacts that are NOT in git, into the deployment
staging folder (docs/internal/do_not_open/requiredforgcp/artifacts/) so that
02_upload_artifacts.sh uploads them to GCS and the VM mounts them at /artifacts.

Artifacts pulled:
  1. Yield LightGBM  — Kaggle KERNEL output:  tanmay23f1/yield-v2-1-7
       -> artifacts/yield/lightgbm_tuned.txt (+ any sibling saved-model files)
       CLI equivalent: kaggle kernels output tanmay23f1/yield-v2-1-7 -p <dest>
  2. IEG checkpoint  — Kaggle DATASET:        lokeshtiwariiitm/ieg-model
       -> artifacts/ieg/intent_entity_guardrail_model.pt + label_maps.json
       CLI equivalent: kaggle datasets download lokeshtiwariiitm/ieg-model
  3. Vision ViT      — Kaggle DATASET:        iitm21f1003346/vits16-crop-disease
       -> artifacts/vision/p3_full_best.pt + label_to_idx.json
  4. Generator adapter — Google Drive folder (distilled LoRA, gdown id below)
       -> artifacts/generator/best_adapter/adapter_model.safetensors (+ config)

Requirements (run where credentials exist):
    pip install kagglehub kaggle gdown
    Kaggle auth: EITHER ~/.kaggle/kaggle.json  OR env vars
        KAGGLE_USERNAME + KAGGLE_KEY   (KAGGLE_KEY accepts the new KGAT_ tokens)
    Public datasets usually download without auth; keep the token for private ones.

Usage:
    python3 scripts/stage_models_gcp.py                 # stage ALL four
    python3 scripts/stage_models_gcp.py --only ieg      # one of: yield|ieg|vision|adapter

Then:
    cd docs/internal/do_not_open/requiredforgcp && bash 02_upload_artifacts.sh
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "docs" / "internal" / "do_not_open" / "requiredforgcp" / "artifacts"

YIELD_KERNEL = "tanmay23f1/yield-v2-1-7"
IEG_DATASET = "lokeshtiwariiitm/ieg-model"
VISION_DATASET = "iitm21f1003346/vits16-crop-disease"

# Distilled LoRA adapter (same id run_e2e_eval.load_generator uses)
ADAPTER_GDRIVE_ID = "1nAbR9-hyba_vUhPE68Idet-oEK4cNA1W"
ADAPTER_REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")

# Yield files we care about (anything else in the kernel output is ignored)
YIELD_WANTED = ["lightgbm_tuned.txt"]


# --------------------------------------------------------------------------
def _kernel_output_via_cli(handle: str, dest: Path) -> bool:
    """kaggle kernels output <handle> -p <dest>"""
    try:
        r = subprocess.run(
            ["kaggle", "kernels", "output", handle, "-p", str(dest)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            print(f"[kaggle cli] kernel output OK:\n{r.stdout.strip()[:400]}")
            return True
        print(f"[kaggle cli] failed ({r.stderr.strip()[:300]})")
    except FileNotFoundError:
        print("[kaggle cli] not installed")
    except subprocess.TimeoutExpired:
        print("[kaggle cli] timed out")
    return False


def _kernel_output_via_hub(handle: str) -> "Path | None":
    try:
        import kagglehub
    except ImportError:
        print("[kagglehub] not installed (pip install kagglehub)")
        return None
    try:
        path = Path(kagglehub.kernel_output(handle))
        print(f"[kagglehub] kernel output at {path}")
        return path
    except Exception as e:
        print(f"[kagglehub] kernel download failed: {e}")
        return None


def _dataset_via_hub(handle: str) -> "Path | None":
    try:
        import kagglehub
    except ImportError:
        print("[kagglehub] not installed (pip install kagglehub)")
        return None
    try:
        path = Path(kagglehub.dataset_download(handle))
        print(f"[kagglehub] dataset at {path}")
        return path
    except Exception as e:
        print(f"[kagglehub] dataset download failed: {e}")
        return None


def _dataset_via_cli(handle: str, dest: Path) -> bool:
    """kaggle datasets download <handle> --unzip -p <dest>"""
    try:
        r = subprocess.run(
            ["kaggle", "datasets", "download", handle, "--unzip", "-p", str(dest)],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode == 0:
            print(f"[kaggle cli] dataset OK:\n{r.stdout.strip()[:400]}")
            return True
        print(f"[kaggle cli] failed ({r.stderr.strip()[:300]})")
    except FileNotFoundError:
        print("[kaggle cli] not installed")
    return False


# --------------------------------------------------------------------------
def stage_yield() -> bool:
    out_dir = ARTIFACTS / "yield"
    src = tempfile.mkdtemp(prefix="fv_yield_")
    src_path = Path(src)

    if not (_kernel_output_via_cli(YIELD_KERNEL, src_path) or
            (p := _kernel_output_via_hub(YIELD_KERNEL)) and _copytree(p, src_path)):
        print("[yield] FAILED to fetch kernel output")
        return False

    staged = []
    for name in YIELD_WANTED:
        hits = list(src_path.rglob(name))
        if not hits:
            print(f"[yield] WARNING: '{name}' not found in kernel output. Contents:")
            for f in sorted(src_path.rglob("*")):
                if f.is_file():
                    print(f"         {f.relative_to(src_path)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / name
        shutil.copy2(hits[0], dst)
        staged.append(dst)
        print(f"[yield] staged {name} ({dst.stat().st_size/1024:.0f} KB)")

    if not staged:
        return False
    print(f"[yield] OK -> {out_dir.relative_to(ROOT)}")
    return True


def _copytree(src: Path, dst: Path) -> bool:
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return True
    except Exception as e:
        print(f"[stage] copy failed: {e}")
        return False


def stage_ieg() -> bool:
    out_dir = ARTIFACTS / "ieg"
    src = tempfile.mkdtemp(prefix="fv_ieg_")
    src_path = Path(src)

    hub = _dataset_via_hub(IEG_DATASET)
    if hub and any(hub.rglob("*.pt")):
        src_path = hub
    elif not _dataset_via_cli(IEG_DATASET, src_path):
        print("[ieg] FAILED to fetch dataset")
        return False

    pts = list(src_path.rglob("*.pt"))
    if not pts:
        print(f"[ieg] ERROR: no .pt checkpoint in dataset. Contents:")
        for f in sorted(src_path.rglob("*")):
            if f.is_file():
                print(f"       {f.relative_to(src_path)}")
        return False
    # Prefer the adamw run (matches previous convention) else first .pt
    ckpt = next((p for p in pts if "adamw" in p.name.lower()), pts[0])

    labels = next(iter(src_path.rglob("label_maps.json")), None)

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_ckpt = out_dir / "intent_entity_guardrail_model.pt"
    shutil.copy2(ckpt, dst_ckpt)
    print(f"[ieg] staged {ckpt.name} ({dst_ckpt.stat().st_size/1e6:.1f} MB)")

    if labels is None:
        print("[ieg] ERROR: label_maps.json missing — the gateway reads "
              "'model_name' from it to build the backbone. Aborting stage.")
        return False
    dst_labels = out_dir / "label_maps.json"
    shutil.copy2(labels, dst_labels)

    # Validate the label maps drive the NEW backbone
    import json
    lm = json.load(open(dst_labels, encoding="utf-8"))
    model_name = lm.get("model_name", "<missing>")
    intents = lm.get("intent_classes", [])
    print(f"[ieg] label_maps: model_name={model_name!r}, "
          f"{len(intents)} intent classes, "
          f"{len(lm.get('ner_labels', []))} NER tags")
    if model_name == "<missing>":
        print("[ieg] WARNING: no 'model_name' key — gateway will default to "
              "distilbert-base-multilingual-cased. Injecting 'l3cube-pune/hing-mbert-mixed' as requested.")
        lm["model_name"] = "l3cube-pune/hing-mbert-mixed"
        model_name = lm["model_name"]
        with open(dst_labels, "w", encoding="utf-8") as f:
            json.dump(lm, f, indent=2)
            
    if "hing-mbert" in model_name:
        print("[ieg] ✓ new hing-mbert backbone confirmed")

    print(f"[ieg] OK -> {out_dir.relative_to(ROOT)}")
    return True


def stage_vision() -> bool:
    out_dir = ARTIFACTS / "vision"
    hub = _dataset_via_hub(VISION_DATASET)
    src_path = None
    if hub and (list(hub.rglob("*.pt")) or list(hub.rglob("*.pth"))):
        src_path = hub
    else:
        tmp = Path(tempfile.mkdtemp(prefix="fv_vision_"))
        if not _dataset_via_cli(VISION_DATASET, tmp):
            print("[vision] FAILED to fetch dataset")
            return False
        src_path = tmp

    pts = list(src_path.rglob("*.pt")) + list(src_path.rglob("*.pth"))
    ckpt = next((p for p in pts if "p3_full_best" in p.name), pts[0] if pts else None)
    if ckpt is None:
        print("[vision] ERROR: no .pt/.pth checkpoint found. Contents:")
        for f in sorted(src_path.rglob("*")):
            if f.is_file():
                print(f"       {f.relative_to(src_path)}")
        return False

    labels = next(iter(src_path.rglob("label_to_idx.json")), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "p3_full_best.pt"
    shutil.copy2(ckpt, dst)
    print(f"[vision] staged p3_full_best.pt ({dst.stat().st_size/1e6:.1f} MB)")

    if labels is None:
        print("[vision] ERROR: label_to_idx.json missing — the gateway's "
              "vision.py needs it to map class indices.")
        return False
    shutil.copy2(labels, out_dir / "label_to_idx.json")
    n_classes = len(json.load(open(out_dir / "label_to_idx.json", encoding="utf-8")))
    print(f"[vision] staged label_to_idx.json ({n_classes} classes)")
    print(f"[vision] OK -> {out_dir.relative_to(ROOT)}")
    return True


def stage_adapter() -> bool:
    """Distilled LoRA adapter from Google Drive -> artifacts/generator/best_adapter."""
    try:
        import gdown
    except ImportError:
        print("[adapter] gdown not installed — pip install gdown")
        return False

    out_dir = ARTIFACTS / "generator" / "best_adapter"
    already = out_dir / ADAPTER_REQUIRED_FILES[0]
    if already.exists():
        print(f"[adapter] already staged: {already}")
        _check_adapter_config(out_dir)
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[adapter] downloading Drive folder {ADAPTER_GDRIVE_ID} …")
    try:
        import inspect
        sig = inspect.signature(gdown.download_folder)
        kwargs = dict(id=ADAPTER_GDRIVE_ID, output=str(out_dir), quiet=False,
                      use_cookies=False)
        if "remaining_ok" in sig.parameters:
            kwargs["remaining_ok"] = True
        gdown.download_folder(**kwargs)
    except Exception as e:
        print(f"[adapter] download failed: {type(e).__name__}: {str(e)[:200]}")
        return False

    # gdown may nest files one level deep — flatten
    hit = next(out_dir.rglob(ADAPTER_REQUIRED_FILES[0]), None)
    if hit and hit.parent != out_dir:
        for f in hit.parent.iterdir():
            shutil.move(str(f), str(out_dir / f.name))

    ok = all((out_dir / f).exists() for f in ADAPTER_REQUIRED_FILES)
    if not ok:
        print(f"[adapter] WARNING: expected {[ADAPTER_REQUIRED_FILES]} under "
              f"{out_dir}; contents:")
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                print(f"         {f.relative_to(out_dir)}")
        return (out_dir / ADAPTER_REQUIRED_FILES[0]).exists()

    _check_adapter_config(out_dir)
    print(f"[adapter] OK -> {out_dir.relative_to(ROOT)}")
    return True


def _check_adapter_config(out_dir: Path):
    cfg = out_dir / "adapter_config.json"
    if not cfg.exists():
        return
    try:
        c = json.load(open(cfg, encoding="utf-8"))
        print(f"[adapter] base_model={c.get('base_model_name_or_path')!r}, "
              f"r={c.get('r')}, targets={len(c.get('target_modules', []))} modules")
    except Exception:
        pass


def _export_kaggle_auth_from_env_file():
    """Load KAGGLE_USERNAME/KAGGLE_KEY from the deployment .env (gitignored)
    into os.environ so kaggle CLI + kagglehub pick them up. Never prints keys."""
    env_file = ROOT / "docs" / "internal" / "do_not_open" / "requiredforgcp" / ".env"
    if not env_file.exists():
        return
    wanted = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in ("KAGGLE_USERNAME", "KAGGLE_KEY") and v.strip():
            wanted.setdefault(k, v.strip())
    for k, v in wanted.items():
        os.environ.setdefault(k, v)   # real env vars take precedence
    if wanted:
        print(f"[auth] loaded Kaggle credentials from {env_file.name} "
              f"(user={os.environ.get('KAGGLE_USERNAME') or 'NOT SET'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["yield", "ieg", "vision", "adapter"],
                    default=None)
    args = ap.parse_args()

    _export_kaggle_auth_from_env_file()

    jobs = {
        "yield": stage_yield,
        "ieg": stage_ieg,
        "vision": stage_vision,
        "adapter": stage_adapter,
    }
    todo = {args.only: jobs[args.only]} if args.only else jobs

    results = {}
    for name, fn in todo.items():
        results[name] = fn()

    print("\n" + "=" * 60)
    ok = all(results.values())
    for k, v in results.items():
        print(f"  {'OK ' if v else 'FAIL'}  {k}")
    if ok:
        print("=" * 60)
        print("Final layout check:")
        expected = [
            ARTIFACTS / "ieg" / "intent_entity_guardrail_model.pt",
            ARTIFACTS / "ieg" / "label_maps.json",
            ARTIFACTS / "vision" / "p3_full_best.pt",
            ARTIFACTS / "vision" / "label_to_idx.json",
            ARTIFACTS / "yield" / "lightgbm_tuned.txt",
            ARTIFACTS / "generator" / "best_adapter" / ADAPTER_REQUIRED_FILES[0],
        ]
        missing = [str(p.relative_to(ARTIFACTS)) for p in expected if not p.exists()]
        if missing:
            print(f"  MISSING (stage separately or with --only): {missing}")
            return 1
        print("  all expected artifacts present ✓")
        print("=" * 60)
        print("Next: cd docs/internal/do_not_open/requiredforgcp && bash 02_upload_artifacts.sh")
        print("(qdrant snapshot is staged separately by scripts/stage_new_snapshot_gcp.py)")
        return 0
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
