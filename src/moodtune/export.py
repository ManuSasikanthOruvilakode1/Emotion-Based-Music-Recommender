"""Export a trained checkpoint to a self-describing ONNX file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import torch

from moodtune import __version__
from moodtune.labels import EMOTIONS
from moodtune.model import load_checkpoint


def export_onnx(checkpoint: Path, output: Path, face_size: int, opset: int = 18) -> dict:
    model, ckpt = load_checkpoint(checkpoint)
    s = model.cfg.image_size
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(1, 3, s, s),),
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    meta = {
        "moodtune_version": __version__,
        "backbone": model.cfg.backbone,
        "pretrained_init": model.cfg.pretrained,
        "image_size": s,
        "face_size": face_size,
        "labels": list(EMOTIONS),
        "temperature": float(model.temperature),
        "epoch": ckpt.get("epoch"),
    }
    proto = onnx.load(str(output))
    onnx.checker.check_model(proto)
    del proto.metadata_props[:]
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.save(proto, str(output))
    return meta


def parity_check(checkpoint: Path, onnx_path: Path, batch: np.ndarray) -> dict:
    """Max absolute logit difference and top-1 agreement between PyTorch and ONNX Runtime."""
    import onnxruntime as ort

    model, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        ref = model(torch.from_numpy(batch)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"image": batch})
    return {
        "max_abs_diff": float(np.abs(ref - got).max()),
        "top1_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
    }
