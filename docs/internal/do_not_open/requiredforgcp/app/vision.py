"""Leaf-disease vision = ViT-S/16 (timm), 20 classes, on CPU.

Faithful to vit-train-01.ipynb:
  * model = Net(backbone, head): timm backbone (num_classes=0) + nn.Linear head
  * checkpoint p3_full_best.pt is a dict with key "model" = net.state_dict()
  * eval transform: CenterCrop(224) + Normalize(0.5,0.5,0.5) — the checkpoint's
    native normalization won (Milestone-5 §8). Inputs were 256x256, so we
    Resize->256 then CenterCrop(224) (crop_pct 0.875, matching training).
  * class order = keys of label_to_idx.json sorted by index.
Runs on CPU on purpose (report §13.2) so the GPU stays free for the LLM.
"""
import io
import json

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from . import config as C

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"
IMG = 224
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

_model = None
_tf = None
_classes = None


class Net(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))


def load():
    global _model, _tf, _classes
    l2i = json.load(open(C.VISION_LABELS, encoding="utf-8"))
    _classes = [k for k, _ in sorted(l2i.items(), key=lambda kv: kv[1])]
    nc = len(_classes)

    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    head = nn.Linear(backbone.num_features, nc)
    net = Net(backbone, head)

    st = torch.load(C.VISION_CKPT, map_location="cpu")
    state = st["model"] if isinstance(st, dict) and "model" in st else st
    net.load_state_dict(state)
    net.eval()
    _model = net

    _tf = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])
    print(f"[vision] loaded {nc} classes")
    return _model


@torch.inference_mode()
def predict(image_bytes: bytes, top_k: int = 5) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0)                      # CPU tensor
    probs = torch.softmax(_model(x), dim=1)[0]
    conf, idx = probs.max(0)
    label = _classes[int(idx)]
    crop, _, disease = label.partition("__")
    tv, ti = probs.topk(min(top_k, len(_classes)))
    top = [{"label": _classes[int(i)], "prob": round(float(v), 4)} for v, i in zip(tv, ti)]
    return {
        "label": label,
        "crop": crop,
        "disease": disease.replace("_", " "),
        "confidence": round(float(conf), 4),
        "top_k": top,
        # honest framing per report §14/§17: this is a suggestion, not a diagnosis
        "note": "Lab-trained model; treat as a suggestion, confirm with a local expert.",
    }
