"""Pengaturan bersama proyek EcoLens Kampus."""

from datetime import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DIR = DATA_DIR / "public"
GD_V2_DIR = PUBLIC_DIR / "garbage_v2" / "standardized_256"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

SEED = 42
AMBANG_CONFIDENCE = 0.6

KELAS = (
    "plastik",
    "kertas_kardus",
    "logam",
    "kaca",
    "organik",
    "residu",
)

PEMETAAN_TRASHNET = {
    "plastic": "plastik",
    "paper": "kertas_kardus",
    "cardboard": "kertas_kardus",
    "metal": "logam",
    "glass": "kaca",
    "trash": "residu",
}

PEMETAAN_GD_V2 = {
    "biological": "organik",
    "plastic": "plastik",
    "paper": "kertas_kardus",
    "cardboard": "kertas_kardus",
    "metal": "logam",
    "glass": "kaca",
    "trash": "residu",
}

# Batas jam memakai waktu lokal kampus; kedua ujung rentang termasuk.
RENTANG_WAKTU = {
    "pagi": (time(5, 0), time(10, 59)),
    "siang": (time(11, 0), time(14, 59)),
    "sore": (time(15, 0), time(18, 59)),
}

SKEMA_HASIL_PREDIKSI = (
    "photo_id",
    "file_name",
    "kode_lokasi",
    "tanggal",
    "jam",
    "waktu",
    "label_manual",
    "prediksi_model",
    "confidence",
    "perlu_verifikasi",
    "sumber",
)
