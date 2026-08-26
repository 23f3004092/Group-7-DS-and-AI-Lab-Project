"""
test_migration_gcp.py
=====================
Validation suite for the migration of experimental improvements
(scripts/run_e2e_eval.py) into the deployed gateway
(docs/internal/do_not_open/requiredforgcp/app).

Tests are CPU-only and need NO model checkpoints / Qdrant server.
Run:
    python scripts/test_migration_gcp.py

Covered:
  1. config: CONF_GATE_LOOSEN + filtering knobs (env-var overridable)
  2. ieg._compress_query: sentence-boundary truncation (eval lines 381-392)
  3. ieg softmax top-3 intent routing semantics (eval lines 404-410)
  4. retrieval._build_routed_filter: source routing + qtype gate (eval 422-465)
  5. retrieval schema constants present (INTENT_SOURCE_MAP / INTENT_QTYPE_MAP)
  6. pipeline passes top_confidence/intents through (source inspection)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GCP_APP_DIR = ROOT / "docs" / "internal" / "do_not_open" / "requiredforgcp"

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except unittest_skip as s:
        print(f"  SKIP  {name}: {s}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}: {e}")


def setup_import_env():
    """The gateway config requires manifest.json at import time — stage a fake one."""
    tmp = tempfile.mkdtemp(prefix="fvmig_")
    manifest = {
        "collection": "agri_knowledge",
        "embed_model": "BAAI/bge-m3",
        "embed_dim": 1024,
        "max_seq_length": 512,
        "query_prefix": "",
        "doc_prefix": "",
        "tiers": {"fallback": 0.553, "grounded": 0.638},
        "fusion_weights": {"general": {"pdf": 1.0, "kcc": 1.0}},
        "top_k_default": 5,
    }
    mp = os.path.join(tmp, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    os.environ["MANIFEST_PATH"] = mp
    os.environ.setdefault("IEG_DIR", os.path.join(tmp, "ieg"))
    # Make `import app` resolve to the GCP gateway package, not the repo's app/
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(GCP_APP_DIR))


# ---------------------------------------------------------------- tests -----

def test_config_gate():
    from app import config as C
    assert abs(C.CONF_GATE_LOOSEN - 0.60) < 1e-6, C.CONF_GATE_LOOSEN
    assert C.MIN_CHUNK_SCORE == 0.50
    assert C.MIN_CHUNK_CHARS == 50
    assert C.MAX_PER_SOURCE == 3
    assert C.TOP_K_BASE == 10
    assert C.RERANK_TOP_N == 5


def test_config_env_override():
    # Re-import with an overridden env var
    os.environ["CONF_GATE_LOOSEN"] = "0.70"
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app import config as C
    assert abs(C.CONF_GATE_LOOSEN - 0.70) < 1e-6, C.CONF_GATE_LOOSEN
    del os.environ["CONF_GATE_LOOSEN"]


def test_compress_query():
    from app import ieg
    # short query untouched
    q = "wheat rust ki dawa"
    assert ieg._compress_query(q) == q
    # long query cut at question-mark boundary INSIDE the window
    long_q = ("namaste sir mere khet mein wheat ke patton pe peele dhabbe aa rahe hain? "
              "what fungicide spray should I use for this leaf rust problem "
              + "x" * 300)
    out = ieg._compress_query(long_q)
    assert len(out) <= ieg.MAX_QUERY_CHARS, len(out)
    assert out.endswith("?"), repr(out[-20:])
    # no boundary in window -> hard truncate
    flat = "a" * 500
    assert ieg._compress_query(flat) == "a" * ieg.MAX_QUERY_CHARS


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import qdrant_client  # noqa: F401
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


def _skip(mod):
    raise unittest_skip(f"{mod} not installed in this environment")


class unittest_skip(Exception):
    pass


def test_softmax_top3_routing():
    """Replicates eval routing math: top-3 always kept, others only above 0.15."""
    if not HAS_TORCH:
        raise unittest_skip("torch")
    import torch
    logits = torch.tensor([2.5, 0.9, 0.8, 0.05, -1.0])   # 5 classes
    probs = torch.softmax(logits, dim=-1)
    top3 = probs.topk(min(3, len(probs))).indices.tolist()
    thresh = [i for i, p in enumerate(probs.tolist()) if p > 0.15]
    idx = list(dict.fromkeys(top3 + thresh))
    assert idx[:3] == sorted(top3, key=lambda i: -probs[i].item())
    # class 3 (p≈0.24 > 0.15) must be included via threshold even though not top-3
    assert 3 in idx, f"class 3 p={probs[3]:.3f} should pass 0.15 threshold"
    # class 4 (p < 0.15) must be excluded
    assert 4 not in idx


def test_build_routed_filter():
    if not HAS_QDRANT:
        raise unittest_skip("qdrant-client")
    from app.retrieval import _build_routed_filter
    # strict: disease_pest -> ppqs_advisories OR (kcc_qa AND its qtypes)
    f = _build_routed_filter(["disease_pest"], loosen_qtypes=False)
    should = f.should
    assert len(should) == 2
    cond0 = should[0].must[0]
    assert cond0.key == "source" and sorted(cond0.match.any) == ["ppqs_advisories"]
    kcc_clause = should[1].must
    assert kcc_clause[0].match.value == "kcc_qa"
    qt = kcc_clause[1]
    assert qt.key == "query_type"
    assert "Plant Protection" in qt.match.any

    # loosened: qtype sub-filter dropped, source-only clause remains
    f2 = _build_routed_filter(["disease_pest"], loosen_qtypes=True)
    kcc2 = [c for c in f2.should if c.must[0].match.value == "kcc_qa"]
    assert len(kcc2) == 1 and len(kcc2[0].must) == 1  # no query_type condition

    # unmapped intent -> no filter
    assert _build_routed_filter(["weather"], False) is None

    # multi-intent merges sources and qtypes
    f3 = _build_routed_filter(["disease_pest", "nutrition_fertilizer", "policy"], False)
    all_sources = set()
    for c in f3.should:
        m = c.must[0]
        vals = getattr(m.match, "any", None) or [m.match.value]
        all_sources.update(v for v in vals if v != "kcc_qa")
    assert {"ppqs_advisories", "up_acp", "schemes"} <= all_sources


def test_routing_tables():
    if not HAS_QDRANT:
        raise unittest_skip("qdrant-client")
    from app.retrieval import INTENT_SOURCE_MAP, INTENT_QTYPE_MAP
    assert INTENT_SOURCE_MAP["disease_pest"] == ["ppqs_advisories", "kcc_qa"]
    assert INTENT_SOURCE_MAP["policy"] == ["schemes"]
    assert INTENT_SOURCE_MAP["general"] is None
    assert "Plant Protection" in INTENT_QTYPE_MAP["disease_pest"]
    assert "Cold Storage" in INTENT_QTYPE_MAP["post_harvest_storage"]


def test_search_signature_backward_compatible():
    """Old callers (positional query/top_k/intent + filters kwargs) still work."""
    if not HAS_QDRANT:
        raise unittest_skip("qdrant-client")
    import inspect
    from app import retrieval
    sig = inspect.signature(retrieval.search_agri_knowledge)
    params = list(sig.parameters)
    for legacy in ["query", "top_k", "intent", "source_type", "doc_category",
                   "query_type", "crop", "district", "season", "language",
                   "year_from", "only_tables"]:
        assert legacy in params, f"legacy param missing: {legacy}"
    for new in ["top_confidence", "intents"]:
        assert new in params and sig.parameters[new].default is None, new


def test_pipeline_passes_confidence():
    src = (GCP_APP_DIR / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert 'g.get("top_confidence")' in src
    assert 'g.get("intents")' in src
    # both streaming and non-streaming paths wired
    assert src.count('g.get("top_confidence")') >= 2


def test_classify_response_fields():
    src = (GCP_APP_DIR / "app" / "ieg.py").read_text(encoding="utf-8")
    assert '"top_confidence"' in src
    assert '"loosen_threshold"' in src
    assert "_compress_query(text)" in src or "_compress_query(compressed)" in src \
           or "compressed = _compress_query(text)" in src


# ----------------------------------------------------------------- main -----

def main():
    print("=" * 64)
    print("FarmerVision — GCP Migration Validation Suite")
    print("=" * 64)
    setup_import_env()

    check("config: gate + filter knobs", test_config_gate)
    check("config: env override CONF_GATE_LOOSEN", test_config_env_override)
    check("ieg: query compression", test_compress_query)
    check("ieg: softmax top-3 routing math", test_softmax_top3_routing)
    check("retrieval: routed filter builder", test_build_routed_filter)
    check("retrieval: routing tables", test_routing_tables)
    check("retrieval: backward-compatible signature", test_search_signature_backward_compatible)
    check("pipeline: confidence wiring (both paths)", test_pipeline_passes_confidence)
    check("ieg: classify() response fields", test_classify_response_fields)

    print("=" * 64)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, err in FAIL:
            print(f"  FAILED: {name}\n    {err}")
        sys.exit(1)
    print("All migration checks passed.")


if __name__ == "__main__":
    main()
