"""Klasifikasikan foto audit kampus dan evaluasi label manual yang tersedia."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2

from src.config import (
    AMBANG_CONFIDENCE,
    KELAS,
    MODEL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORT_DIR,
    SKEMA_HASIL_PREDIKSI,
)
from src.train import RATA_RATA_IMAGENET, STD_IMAGENET, UKURAN_GAMBAR
from src.utils import parse_nama_foto_audit, tentukan_waktu


EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetAudit(Dataset):
    """Membaca foto audit tanpa mempertahankan metadata EXIF."""

    def __init__(self, foto: list[dict[str, object]], transformasi):
        self.foto = foto
        self.transformasi = transformasi

    def __len__(self) -> int:
        return len(self.foto)

    def __getitem__(self, indeks: int):
        foto = self.foto[indeks]
        path = foto["path"]
        try:
            with Image.open(path) as gambar:
                gambar.getexif().clear()
                gambar_rgb = gambar.convert("RGB")
        except (OSError, UnidentifiedImageError) as kesalahan:
            raise ValueError(f"Tidak dapat membaca foto audit: {path}") from kesalahan
        return self.transformasi(gambar_rgb)


def pilih_device(nama_device: str) -> torch.device:
    if nama_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA diminta, tetapi tidak tersedia pada perangkat ini.")
    if nama_device == "cpu":
        return torch.device("cpu")
    if nama_device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cari_foto_audit(folder_audit: Path) -> list[dict[str, object]]:
    """Temukan dan validasi nama/lokasi semua foto audit."""
    if not folder_audit.is_dir():
        raise FileNotFoundError(
            f"Folder foto audit tidak ditemukan: {folder_audit}. "
            r"Siapkan foto pada data\raw\foto_audit\<kode_lokasi>\."
        )

    foto = []
    nama_id = set()
    for path in sorted(folder_audit.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in EKSTENSI_GAMBAR:
            continue
        metadata = parse_nama_foto_audit(path.name)
        if metadata is None:
            raise ValueError(f"Nama foto audit tidak sesuai pola: {path.name}")

        bagian_rel = path.relative_to(folder_audit).parts
        if not bagian_rel or bagian_rel[0].upper() != metadata.kode_lokasi:
            raise ValueError(
                f"Kode lokasi folder tidak cocok dengan nama file: {path}"
            )

        photo_id = path.relative_to(folder_audit).with_suffix("").as_posix()
        if photo_id in nama_id:
            raise ValueError(f"photo_id duplikat pada foto audit: {photo_id}")
        nama_id.add(photo_id)
        foto.append(
            {
                "path": path,
                "photo_id": photo_id,
                "file_name": path.name,
                "kode_lokasi": metadata.kode_lokasi,
                "tanggal": (
                    f"{metadata.tanggal[:4]}-{metadata.tanggal[4:6]}-"
                    f"{metadata.tanggal[6:8]}"
                ),
                "jam": f"{metadata.jam[:2]}:{metadata.jam[2:]}",
                "waktu": tentukan_waktu(metadata.jam),
            }
        )

    if not foto:
        raise FileNotFoundError(f"Tidak ada foto audit yang didukung di: {folder_audit}")
    return foto


def baca_label_manual(path_label: Path, foto: list[dict[str, object]]) -> dict[str, str]:
    """Baca label manual yang terhubung dengan foto audit yang ditemukan."""
    if not path_label.is_file():
        return {}

    nama_foto = {str(item["file_name"]) for item in foto}
    label = {}
    with path_label.open("r", newline="", encoding="utf-8-sig") as file_csv:
        pembaca = csv.DictReader(file_csv)
        kolom_wajib = {"file_name", "label_manual"}
        if not kolom_wajib.issubset(pembaca.fieldnames or []):
            raise ValueError("label_audit.csv harus memiliki kolom file_name,label_manual.")
        for baris in pembaca:
            file_name = (baris.get("file_name") or "").strip()
            label_manual = (baris.get("label_manual") or "").strip()
            if not file_name and not label_manual:
                continue
            if file_name not in nama_foto:
                raise ValueError(f"Label merujuk foto audit yang tidak ditemukan: {file_name}")
            if label_manual not in KELAS:
                raise ValueError(
                    f"Label manual tidak dikenal untuk {file_name}: {label_manual}"
                )
            if file_name in label:
                raise ValueError(f"Label manual duplikat untuk foto: {file_name}")
            label[file_name] = label_manual
    return label


def muat_model(path_model: Path, path_label: Path, device: torch.device):
    """Muat model terlatih dan pastikan urutan label sesuai konfigurasi."""
    if not path_model.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {path_model}")
    if not path_label.is_file():
        raise FileNotFoundError(f"Daftar label tidak ditemukan: {path_label}")

    with path_label.open("r", encoding="utf-8") as file_json:
        label_file = json.load(file_json)
    labels = [label_file.get(str(indeks)) for indeks in range(len(KELAS))]
    if labels != list(KELAS):
        raise ValueError("Urutan label pada labels.json tidak cocok dengan config.py.")

    checkpoint = torch.load(path_model, map_location=device, weights_only=True)
    if checkpoint.get("kelas") != list(KELAS):
        raise ValueError("Urutan kelas pada checkpoint tidak cocok dengan config.py.")
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(KELAS))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def hitung_metrik_audit(
    labels_manual: list[str], prediksi: list[str]
) -> dict[str, object]:
    indeks = list(range(len(KELAS)))
    label_ke_indeks = {label: nomor for nomor, label in enumerate(KELAS)}
    aktual = [label_ke_indeks[label] for label in labels_manual]
    hasil = [label_ke_indeks[label] for label in prediksi]
    rincian = classification_report(
        aktual,
        hasil,
        labels=indeks,
        target_names=list(KELAS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "jumlah_foto_dengan_label_manual": len(aktual),
        "accuracy": float(accuracy_score(aktual, hasil)),
        "per_kelas": {
            kelas: {
                "precision": float(rincian[kelas]["precision"]),
                "recall": float(rincian[kelas]["recall"]),
                "f1": float(rincian[kelas]["f1-score"]),
                "support": int(rincian[kelas]["support"]),
            }
            for kelas in KELAS
        },
        "macro_avg": {
            "precision": float(rincian["macro avg"]["precision"]),
            "recall": float(rincian["macro avg"]["recall"]),
            "f1": float(rincian["macro avg"]["f1-score"]),
        },
        "confusion_matrix": confusion_matrix(aktual, hasil, labels=indeks).tolist(),
    }


def simpan_matriks_audit(metrik: dict[str, object]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matriks = np.asarray(metrik["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    gambar = ax.imshow(matriks, cmap="Blues")
    ax.set_title("Confusion matrix foto audit berlabel manual")
    ax.set_xticks(range(len(KELAS)), labels=KELAS, rotation=45, ha="right")
    ax.set_yticks(range(len(KELAS)), labels=KELAS)
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Label manual")
    for baris in range(len(KELAS)):
        for kolom in range(len(KELAS)):
            ax.text(kolom, baris, str(matriks[baris, kolom]), ha="center", va="center")
    fig.colorbar(gambar, ax=ax)
    fig.tight_layout()
    tujuan = REPORT_DIR / "confusion_matrix_audit.png"
    fig.savefig(tujuan, dpi=160)
    plt.close(fig)
    return tujuan


def jalankan(args: argparse.Namespace) -> Path:
    device = pilih_device(args.device)
    foto = cari_foto_audit(args.audit_dir)
    label_manual = baca_label_manual(args.label_file, foto)
    model = muat_model(args.model, args.labels, device)

    metric_latih_path = REPORT_DIR / "metrics_latih.json"
    if metric_latih_path.is_file():
        with metric_latih_path.open("r", encoding="utf-8") as file_json:
            metrik_latih = json.load(file_json)
        if not metrik_latih.get("model_dilatih_dengan_foto_kampus", False):
            print(
                "PERINGATAN: model belum dilatih dengan foto kampus; "
                "tinjau hasil prediksi secara hati-hati."
            )
    else:
        print("PERINGATAN: laporan pelatihan tidak ditemukan; asal pelatihan model belum terverifikasi.")

    transformasi = transforms.Compose(
        [
            transforms.Resize((UKURAN_GAMBAR, UKURAN_GAMBAR)),
            transforms.ToTensor(),
            transforms.Normalize(RATA_RATA_IMAGENET, STD_IMAGENET),
        ]
    )
    loader = DataLoader(
        DatasetAudit(foto, transformasi),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prediksi_indeks = []
    confidence = []
    with torch.inference_mode():
        for batch in loader:
            probabilitas = torch.softmax(model(batch.to(device)), dim=1)
            nilai_confidence, indeks = probabilitas.max(dim=1)
            prediksi_indeks.extend(indeks.cpu().tolist())
            confidence.extend(nilai_confidence.cpu().tolist())

    hasil = []
    for item, indeks, yakin in zip(foto, prediksi_indeks, confidence):
        kelas_prediksi = KELAS[indeks]
        hasil.append(
            {
                "photo_id": item["photo_id"],
                "file_name": item["file_name"],
                "kode_lokasi": item["kode_lokasi"],
                "tanggal": item["tanggal"],
                "jam": item["jam"],
                "waktu": item["waktu"],
                "label_manual": label_manual.get(str(item["file_name"]), ""),
                "prediksi_model": kelas_prediksi,
                "confidence": f"{yakin:.6f}",
                "perlu_verifikasi": str(yakin < AMBANG_CONFIDENCE),
                "sumber": "kampus",
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path_hasil = PROCESSED_DIR / "hasil_prediksi.csv"
    with path_hasil.open("w", newline="", encoding="utf-8") as file_csv:
        penulis = csv.DictWriter(file_csv, fieldnames=SKEMA_HASIL_PREDIKSI)
        penulis.writeheader()
        penulis.writerows(hasil)

    print(f"Jumlah foto audit diprediksi: {len(hasil)}")
    print(f"Jumlah foto perlu verifikasi: {sum(item['perlu_verifikasi'] == 'True' for item in hasil)}")
    print(f"Hasil prediksi tersimpan: {path_hasil}")

    if label_manual:
        hasil_berlabel = [
            item for item in hasil if item["label_manual"] in KELAS
        ]
        metrik = hitung_metrik_audit(
            [item["label_manual"] for item in hasil_berlabel],
            [item["prediksi_model"] for item in hasil_berlabel],
        )
        metrik["kelas"] = list(KELAS)
        metrik["catatan"] = (
            "Evaluasi foto kampus menggunakan hanya foto yang memiliki label manual; "
            "metrik dataset publik disimpan terpisah."
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path_metrik = REPORT_DIR / "metrics_audit.json"
        with path_metrik.open("w", encoding="utf-8") as file_json:
            json.dump(metrik, file_json, indent=2, ensure_ascii=False)
        path_matriks = simpan_matriks_audit(metrik)
        print(f"Accuracy audit berlabel: {metrik['accuracy']:.4f}")
        print(f"Metrik audit tersimpan: {path_metrik}")
        print(f"Confusion matrix audit tersimpan: {path_matriks}")
    else:
        print("Label audit belum tersedia; metrik akurasi foto kampus tidak dihitung.")
    return path_hasil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=RAW_DIR / "foto_audit")
    parser.add_argument("--label-file", type=Path, default=RAW_DIR / "label_audit.csv")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "ecolens_mobilenetv2.pt")
    parser.add_argument("--labels", type=Path, default=MODEL_DIR / "labels.json")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Perangkat; auto memilih CUDA bila tersedia, selain itu CPU.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size harus minimal 1.")

    try:
        jalankan(args)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
