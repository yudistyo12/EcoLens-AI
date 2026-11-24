"""Ukur waktu inferensi ONNX pada split uji Kaggle tanpa menghitung preprocessing."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import PROJECT_ROOT
from src.train import RATA_RATA_IMAGENET, STD_IMAGENET, UKURAN_GAMBAR


NAMA_MODEL = "ecolens_mobilenetv2.onnx"
NAMA_MANIFEST = "data/processed/manifest_latih.csv"


def sha256_berkas(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as berkas:
        for bagian in iter(lambda: berkas.read(1024 * 1024), b""):
            hasher.update(bagian)
    return hasher.hexdigest()


def baca_tensor_uji(root: Path) -> list[np.ndarray]:
    path_manifest = root / NAMA_MANIFEST
    with path_manifest.open("r", newline="", encoding="utf-8") as file_csv:
        baris_uji = [
            item
            for item in csv.DictReader(file_csv)
            if item.get("split") == "test"
        ]
    if not baris_uji:
        raise ValueError(f"Tidak ada foto uji pada manifest: {path_manifest}")

    ukuran = getattr(Image, "Resampling", Image)
    rata_rata = np.asarray(RATA_RATA_IMAGENET, dtype=np.float32)
    standar_deviasi = np.asarray(STD_IMAGENET, dtype=np.float32)
    tensor_uji = []
    for item in baris_uji:
        path_foto = root / item["path"]
        if not path_foto.is_file():
            raise FileNotFoundError(f"Foto uji tidak ditemukan: {path_foto}")
        with Image.open(path_foto) as gambar:
            gambar.getexif().clear()
            gambar_rgb = gambar.convert("RGB").resize(
                (UKURAN_GAMBAR, UKURAN_GAMBAR), ukuran.BILINEAR
            )
            array = np.asarray(gambar_rgb, dtype=np.float32) / 255.0
        array = (array - rata_rata) / standar_deviasi
        tensor_uji.append(np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32))
    return tensor_uji


def ukur(root: Path, tujuan: Path, jumlah_pemanasan: int = 10) -> dict[str, object]:
    path_model = root / "models" / NAMA_MODEL
    path_manifest = root / NAMA_MANIFEST
    if not path_model.is_file():
        raise FileNotFoundError(f"Model ONNX tidak ditemukan: {path_model}")
    if not path_manifest.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {path_manifest}")
    if jumlah_pemanasan < 0:
        raise ValueError("Jumlah foto pemanasan tidak boleh negatif.")

    tensor_uji = baca_tensor_uji(root)
    sesi = ort.InferenceSession(
        str(path_model), providers=["CPUExecutionProvider"]
    )
    nama_input = sesi.get_inputs()[0].name
    jumlah_pemanasan_aktual = min(jumlah_pemanasan, len(tensor_uji))
    for tensor in tensor_uji[:jumlah_pemanasan_aktual]:
        sesi.run(None, {nama_input: tensor})

    waktu_ms = []
    for tensor in tensor_uji:
        mulai = time.perf_counter()
        sesi.run(None, {nama_input: tensor})
        waktu_ms.append((time.perf_counter() - mulai) * 1000)

    laporan = {
        "format": "ecolens_onnx_benchmark_v1",
        "waktu_pengukuran": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider": sesi.get_providers()[0],
        "prosesor": platform.processor(),
        "jumlah_foto_uji": len(tensor_uji),
        "jumlah_foto_pemanasan": jumlah_pemanasan_aktual,
        "rata_rata_ms": statistics.mean(waktu_ms),
        "median_ms": statistics.median(waktu_ms),
        "p95_ms": float(np.percentile(waktu_ms, 95)),
        "cakupan_waktu": (
            "Hanya sesi inferensi ONNX; dekode, resize, dan normalisasi gambar "
            "tidak dihitung."
        ),
        "model_bytes": path_model.stat().st_size,
        "model_sha256": sha256_berkas(path_model),
        "manifest_sha256": sha256_berkas(path_manifest),
    }
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    with tujuan.open("w", encoding="utf-8") as file_json:
        json.dump(laporan, file_json, indent=2, ensure_ascii=False)
    return laporan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "reports/benchmark_onnx.json"
    laporan = ukur(root, output, args.warmup)
    print(
        f"Benchmark ONNX: {laporan['jumlah_foto_uji']} foto, "
        f"rata-rata {laporan['rata_rata_ms']:.3f} ms; "
        f"laporan tersimpan di {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
