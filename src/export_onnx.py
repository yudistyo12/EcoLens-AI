"""Ekspor checkpoint MobileNetV2 ke ONNX untuk inferensi dashboard CPU."""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from src.config import KELAS, MODEL_DIR


UKURAN_INPUT = 224


def ekspor(checkpoint_path: Path, output_path: Path) -> Path:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint model tidak ditemukan: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kelas") != list(KELAS):
        raise ValueError("Urutan kelas checkpoint tidak cocok dengan config.py.")

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(KELAS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dummy = torch.zeros((1, 3, UKURAN_INPUT, UKURAN_INPUT), dtype=torch.float32)
    torch.onnx.export(
        model,
        input_dummy,
        output_path,
        input_names=["gambar"],
        output_names=["logits"],
        dynamic_axes={"gambar": {0: "jumlah_foto"}, "logits": {0: "jumlah_foto"}},
        opset_version=17,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(output_path))
    sesi = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    hasil_onnx = sesi.run(None, {"gambar": input_dummy.numpy()})[0]
    with torch.inference_mode():
        hasil_torch = model(input_dummy).numpy()
    np.testing.assert_allclose(hasil_onnx, hasil_torch, rtol=1e-3, atol=1e-4)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_DIR / "ecolens_mobilenetv2.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=MODEL_DIR / "ecolens_mobilenetv2.onnx"
    )
    args = parser.parse_args()
    try:
        hasil = ekspor(args.checkpoint, args.output)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2
    print(f"Model ONNX tersimpan dan valid: {hasil}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
