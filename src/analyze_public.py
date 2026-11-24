"""Buat analisis dan kandidat rekomendasi dari dataset Kaggle serta metrik model."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import KELAS, PEMETAAN_GD_V2, PROJECT_ROOT, REPORT_DIR


EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cari_dataset(root: Path) -> Path:
    pilihan = (
        root / "data/public/garbage_v2/standardized_256",
        root / "dataset/standardized_256",
    )
    for folder in pilihan:
        if folder.is_dir():
            return folder
    raise FileNotFoundError(
        "Folder standardized_256 tidak ditemukan pada lokasi dataset proyek."
    )


def hitung_foto(folder: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in EKSTENSI_GAMBAR
        for path in folder.rglob("*")
    )


def baca_manifest(root: Path) -> list[dict[str, str]]:
    path_manifest = root / "data/processed/manifest_latih.csv"
    if not path_manifest.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {path_manifest}")
    with path_manifest.open("r", newline="", encoding="utf-8") as file_csv:
        pembaca = csv.DictReader(file_csv)
        kolom = {"path", "kelas", "sumber", "group_id", "split"}
        if not kolom.issubset(pembaca.fieldnames or []):
            raise ValueError("Kolom manifest latih tidak sesuai skema.")
        baris = list(pembaca)

    for item in baris:
        if item["sumber"] != "gd_v2":
            raise ValueError(
                "Analisis Kaggle hanya menerima baris sumber=gd_v2 pada manifest."
            )
        if item["kelas"] not in KELAS or item["split"] not in {"train", "val", "test"}:
            raise ValueError("Kelas atau split tidak valid pada manifest.")
        if not (root / item["path"]).is_file():
            raise FileNotFoundError(f"Foto manifest tidak ditemukan: {item['path']}")
    if not baris:
        raise ValueError("Manifest Kaggle tidak berisi foto.")
    return baris


def baca_metrik(root: Path) -> dict:
    path_metrik = root / "reports/metrics_latih.json"
    if not path_metrik.is_file():
        raise FileNotFoundError(f"Laporan metrik tidak ditemukan: {path_metrik}")
    with path_metrik.open("r", encoding="utf-8") as file_json:
        laporan = json.load(file_json)
    publik = laporan.get("evaluasi_dataset_publik")
    if not publik or not publik.get("per_kelas") or not publik.get("confusion_matrix"):
        raise ValueError("Metrik publik belum lengkap untuk dianalisis.")
    if any(kelas not in publik["per_kelas"] for kelas in KELAS):
        raise ValueError("Laporan metrik tidak mencakup semua kelas.")
    return laporan


def buat_analisis(root: Path = PROJECT_ROOT) -> dict[str, object]:
    folder_dataset = cari_dataset(root)
    baris_manifest = baca_manifest(root)
    laporan_model = baca_metrik(root)
    metrik = laporan_model["evaluasi_dataset_publik"]

    hitungan_asal = {}
    hitungan_pemetaan = Counter()
    for folder in sorted(folder_dataset.iterdir(), key=lambda item: item.name.lower()):
        if not folder.is_dir():
            continue
        nama_asal = folder.name.lower()
        jumlah = hitung_foto(folder)
        hitungan_asal[nama_asal] = jumlah
        kelas_tujuan = PEMETAAN_GD_V2.get(nama_asal)
        if kelas_tujuan:
            hitungan_pemetaan[kelas_tujuan] += jumlah

    split_per_kelas: dict[str, Counter] = defaultdict(Counter)
    for item in baris_manifest:
        split_per_kelas[item["kelas"]][item["split"]] += 1

    metrik_kelas = metrik["per_kelas"]
    recall_terendah = min(metrik_kelas[kelas]["recall"] for kelas in KELAS)
    kelas_recall_terendah = sorted(
        kelas
        for kelas in KELAS
        if metrik_kelas[kelas]["recall"] == recall_terendah
    )

    confusion = metrik["confusion_matrix"]
    pasangan_salah = []
    for indeks_asli, kelas_asli in enumerate(KELAS):
        for indeks_prediksi, kelas_prediksi in enumerate(KELAS):
            if indeks_asli != indeks_prediksi and confusion[indeks_asli][indeks_prediksi] > 0:
                pasangan_salah.append(
                    (
                        int(confusion[indeks_asli][indeks_prediksi]),
                        kelas_asli,
                        kelas_prediksi,
                    )
                )
    if pasangan_salah:
        jumlah_salah_tertinggi = max(item[0] for item in pasangan_salah)
        kebingungan_tertinggi = [
            {
                "label_sebenarnya": kelas_asli,
                "prediksi_model": kelas_prediksi,
                "jumlah_foto": jumlah,
            }
            for jumlah, kelas_asli, kelas_prediksi in sorted(
                pasangan_salah, key=lambda item: (-item[0], item[1], item[2])
            )
            if jumlah == jumlah_salah_tertinggi
        ]
    else:
        kebingungan_tertinggi = []

    total_manifest = len(baris_manifest)
    insight = [
        (
            f"Pada {metrik['jumlah_foto_uji']} foto uji, accuracy model "
            f"{metrik['accuracy']:.1%} dan macro F1 "
            f"{metrik['macro_avg']['f1']:.1%}."
        ),
        (
            f"Recall terendah terdapat pada kelas {', '.join(kelas_recall_terendah)} "
            f"({recall_terendah:.1%})."
        ),
    ]
    if kebingungan_tertinggi:
        salah_tertinggi = kebingungan_tertinggi[0]
        insight.append(
            f"Pasangan salah-klasifikasi terbanyak: label "
            f"{salah_tertinggi['label_sebenarnya']} diprediksi "
            f"{salah_tertinggi['prediksi_model']} "
            f"({salah_tertinggi['jumlah_foto']} foto uji)."
        )
    else:
        insight.append("Tidak ada salah-klasifikasi pada confusion matrix uji.")

    kandidat = []
    if kelas_recall_terendah:
        kandidat.append(
            {
                "judul": "Tinjau kelas dengan recall terendah",
                "dasar": insight[1],
                "usulan_kandidat": (
                    "Periksa contoh salah-klasifikasi kelas tersebut dan uji "
                    "perbaikan data/pelatihan sebelum menetapkan solusi."
                ),
            }
        )
    if kebingungan_tertinggi:
        salah_tertinggi = kebingungan_tertinggi[0]
        kandidat.append(
            {
                "judul": "Tinjau pasangan kelas yang paling sering tertukar",
                "dasar": insight[2],
                "usulan_kandidat": (
                    "Tinjau kemiripan visual dan panduan label pasangan kelas ini, "
                    "lalu bandingkan metrik setelah perbaikan."
                ),
            }
        )

    return {
        "sumber": {
            "nama": "Garbage Classification V2",
            "url": "https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2",
            "lisensi": "MIT (sesuai keterangan sumber dataset)",
            "folder_lokal": folder_dataset.relative_to(root).as_posix(),
            "peran": "Pelatihan dan evaluasi model; bukan data lokasi kampus.",
        },
        "dataset_asal": {
            "jumlah_foto_per_folder": hitungan_asal,
            "jumlah_foto_total": sum(hitungan_asal.values()),
            "pemetaan_per_folder": {
                nama: PEMETAAN_GD_V2.get(nama, "diabaikan")
                for nama in hitungan_asal
            },
            "jumlah_per_kelas_setelah_pemetaan": {
                kelas: hitungan_pemetaan[kelas] for kelas in KELAS
            },
            "kelas_diabaikan": sorted(set(hitungan_asal) - set(PEMETAAN_GD_V2)),
        },
        "manifest_sampel_model": {
            "jumlah_foto": total_manifest,
            "jumlah_per_kelas_dan_split": {
                kelas: {
                    split: split_per_kelas[kelas][split]
                    for split in ("train", "val", "test")
                }
                for kelas in KELAS
            },
            "catatan": (
                "Manifest dibatasi/sampling untuk pelatihan; jumlahnya bukan "
                "prevalensi sampah di lingkungan nyata."
            ),
        },
        "evaluasi_model_publik": metrik,
        "insight_terhitung": insight,
        "kandidat_rekomendasi": kandidat,
        "batasan": (
            "Hasil hanya menjelaskan performa pada dataset publik dan tidak "
            "menunjukkan pola lokasi, waktu, atau akurasi di kampus."
        ),
    }


def simpan_laporan(root: Path, laporan: dict[str, object]) -> tuple[Path, Path, Path]:
    folder_laporan = root / "reports"
    folder_laporan.mkdir(parents=True, exist_ok=True)
    path_json = folder_laporan / "analisis_kaggle.json"
    path_inventory = folder_laporan / "inventaris_gd_v2.csv"
    path_rekomendasi = folder_laporan / "kandidat_rekomendasi.md"

    with path_json.open("w", encoding="utf-8") as file_json:
        json.dump(laporan, file_json, indent=2, ensure_ascii=False)
    with path_inventory.open("w", newline="", encoding="utf-8") as file_csv:
        penulis = csv.DictWriter(
            file_csv, fieldnames=("folder_asal", "kelas_pemetaan", "jumlah_foto")
        )
        penulis.writeheader()
        jumlah_asal = laporan["dataset_asal"]["jumlah_foto_per_folder"]
        for nama_asal, jumlah in jumlah_asal.items():
            penulis.writerow(
                {
                    "folder_asal": nama_asal,
                    "kelas_pemetaan": PEMETAAN_GD_V2.get(nama_asal, "diabaikan"),
                    "jumlah_foto": jumlah,
                }
            )

    with path_rekomendasi.open("w", encoding="utf-8") as file_md:
        file_md.write("# Kandidat Rekomendasi (perlu ditinjau tim)\n\n")
        file_md.write(
            "Kandidat berikut dibuat dari metrik dataset publik. Ini bukan "
            "rekomendasi final dan tidak menjelaskan penataan fasilitas kampus.\n\n"
        )
        for indeks, item in enumerate(laporan["kandidat_rekomendasi"], start=1):
            file_md.write(f"## Kandidat {indeks}: {item['judul']}\n\n")
            file_md.write(f"- Dasar terhitung: {item['dasar']}\n")
            file_md.write(f"- Usulan untuk ditinjau: {item['usulan_kandidat']}\n\n")
        file_md.write(
            "Tim wajib meninjau contoh gambar dan metrik sebelum mengubah "
            "kandidat menjadi rekomendasi final.\n"
        )
    return path_json, path_inventory, path_rekomendasi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        laporan = buat_analisis(args.root.resolve())
        paths = simpan_laporan(args.root.resolve(), laporan)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2

    print(f"Foto sumber lokal terhitung: {laporan['dataset_asal']['jumlah_foto_total']}")
    print(f"Foto manifest: {laporan['manifest_sampel_model']['jumlah_foto']}")
    for nomor, item in enumerate(laporan["insight_terhitung"], start=1):
        print(f"Insight {nomor}: {item}")
    print("Laporan tersimpan:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
