"""Pemeriksaan awal untuk menolak foto yang memuat wajah terdeteksi."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from src.config import MODEL_DIR


MODEL_WAJAH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
UKURAN_INPUT = (320, 320)
AMBANG_WAJAH = 0.7
_KUNCI_DETEKTOR = Lock()


@lru_cache(maxsize=1)
def _muat_detektor(path_model: str):
    try:
        return cv2.FaceDetectorYN.create(
            path_model,
            "",
            UKURAN_INPUT,
            AMBANG_WAJAH,
            0.3,
            5000,
        )
    except cv2.error as kesalahan:
        raise RuntimeError(
            f"Model deteksi wajah tidak dapat dimuat: {kesalahan}"
        ) from kesalahan


def ada_wajah_terdeteksi(
    gambar: Image.Image, path_model: Path = MODEL_WAJAH
) -> bool:
    path_model = Path(path_model)
    if not path_model.is_file():
        raise FileNotFoundError(f"Model deteksi wajah tidak ditemukan: {path_model}")

    piksel_rgb = np.asarray(gambar.convert("RGB"), dtype=np.uint8)
    tinggi, lebar = piksel_rgb.shape[:2]
    if tinggi == 0 or lebar == 0:
        raise ValueError("Gambar kosong dan tidak dapat diperiksa.")

    skala = min(UKURAN_INPUT[0] / lebar, UKURAN_INPUT[1] / tinggi)
    lebar_baru = max(1, round(lebar * skala))
    tinggi_baru = max(1, round(tinggi * skala))
    interpolasi = cv2.INTER_AREA if skala < 1 else cv2.INTER_LINEAR
    gambar_ubah_ukuran = cv2.resize(
        piksel_rgb, (lebar_baru, tinggi_baru), interpolation=interpolasi
    )
    kanvas = np.zeros((*UKURAN_INPUT[::-1], 3), dtype=np.uint8)
    kiri = (UKURAN_INPUT[0] - lebar_baru) // 2
    atas = (UKURAN_INPUT[1] - tinggi_baru) // 2
    kanvas[atas : atas + tinggi_baru, kiri : kiri + lebar_baru] = gambar_ubah_ukuran
    gambar_bgr = cv2.cvtColor(kanvas, cv2.COLOR_RGB2BGR)

    detektor = _muat_detektor(str(path_model.resolve()))
    try:
        with _KUNCI_DETEKTOR:
            detektor.setInputSize(UKURAN_INPUT)
            _, wajah = detektor.detect(gambar_bgr)
    except cv2.error as kesalahan:
        raise RuntimeError(
            f"Pemeriksaan wajah gagal dijalankan: {kesalahan}"
        ) from kesalahan
    return wajah is not None and len(wajah) > 0
