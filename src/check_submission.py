"""Periksa demo klasifikasi Kaggle dan berkas pengumpulan yang perlu disiapkan tim."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.config import KELAS, PROJECT_ROOT, REPORT_DIR


KOLOM_MANIFEST = {"path", "kelas", "sumber", "group_id", "split"}
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tambah(
    daftar: list[dict[str, str]], pemeriksaan: str, status: str, rincian: str
) -> None:
    daftar.append(
        {"pemeriksaan": pemeriksaan, "status": status, "rincian": rincian}
    )


def baca_json(path: Path):
    with path.open("r", encoding="utf-8") as file_json:
        return json.load(file_json)


def periksa_dokumen(root: Path, hasil: list[dict[str, str]]) -> None:
    wajib = (
        "app.py",
        "src/prepare_data.py",
        "src/train.py",
        "src/analyze_public.py",
        "src/export_onnx.py",
        "src/create_summary.py",
        "src/local_history.py",
        ".streamlit/config.toml",
        "requirements.txt",
        "docs/DATA_SOURCES.md",
        "docs/DEPLOYMENT.md",
        "docs/AI_DISCLOSURE_LOG.md",
        "content/rekomendasi.md",
        "content/NASKAH_VIDEO.md",
    )
    hilang = [nama for nama in wajib if not (root / nama).is_file()]
    if hilang:
        tambah(hasil, "Kode dan dokumentasi", "TERBLOKIR", "Berkas hilang: " + ", ".join(hilang))
    else:
        tambah(
            hasil,
            "Kode dan dokumentasi",
            "LULUS",
            "Kode, sumber data, deployment, AI disclosure, dan template tersedia.",
        )

    sample_path = root / "data/sample/hasil_prediksi_CONTOH.csv"
    if sample_path.is_file():
        with sample_path.open("r", newline="", encoding="utf-8") as file_csv:
            sample = list(csv.DictReader(file_csv))
        valid = bool(sample) and all(
            item.get("sumber") == "CONTOH_SINTETIK" for item in sample
        )
        tambah(
            hasil,
            "File contoh terisolasi",
            "LULUS" if valid else "TERBLOKIR",
            "File contoh berlabel sintetik dan tidak menjadi sumber insight dashboard."
            if valid
            else "Ada baris contoh tanpa label CONTOH_SINTETIK.",
        )


def periksa_manifest(
    root: Path, hasil: list[dict[str, str]]
) -> tuple[list[dict[str, str]], Counter]:
    path_manifest = root / "data/processed/manifest_latih.csv"
    if not path_manifest.is_file():
        tambah(hasil, "Manifest Kaggle", "TERBLOKIR", "Jalankan src.prepare_data.")
        return [], Counter()

    with path_manifest.open("r", newline="", encoding="utf-8") as file_csv:
        pembaca = csv.DictReader(file_csv)
        if not KOLOM_MANIFEST.issubset(pembaca.fieldnames or []):
            tambah(hasil, "Manifest Kaggle", "TERBLOKIR", "Skema manifest tidak sesuai.")
            return [], Counter()
        baris = list(pembaca)

    jumlah_split = Counter()
    split_grup: dict[str, set[str]] = defaultdict(set)
    berkas_per_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    masalah = []
    for item in baris:
        if item["sumber"] != "gd_v2":
            masalah.append("sumber selain gd_v2")
        if item["kelas"] not in KELAS:
            masalah.append(f"kelas tidak dikenal: {item['kelas']}")
        if item["split"] not in {"train", "val", "test"}:
            masalah.append(f"split tidak dikenal: {item['split']}")
        if item["group_id"] and item["split"]:
            split_grup[item["group_id"]].add(item["split"])
        jumlah_split[item["split"]] += 1
        path_foto = root / item["path"]
        if not path_foto.is_file():
            masalah.append(f"path hilang: {item['path']}")
        else:
            try:
                hash_foto = hashlib.sha256(path_foto.read_bytes()).hexdigest()
            except OSError:
                masalah.append(f"file tidak dapat dibaca: {item['path']}")
            else:
                berkas_per_hash[hash_foto].append((item["path"], item["split"]))
    bocor = sum(len(split) > 1 for split in split_grup.values())
    if bocor:
        masalah.append(f"{bocor} group_id lintas split")
    duplikat = [
        baris_duplikat
        for baris_duplikat in berkas_per_hash.values()
        if len(baris_duplikat) > 1
    ]
    jumlah_duplikat = sum(len(baris_duplikat) - 1 for baris_duplikat in duplikat)
    duplikat_lintas_split = sum(
        len({split for _, split in baris_duplikat}) > 1
        for baris_duplikat in duplikat
    )
    if jumlah_duplikat:
        masalah.append(f"{jumlah_duplikat} file dengan isi gambar duplikat")
    if duplikat_lintas_split:
        masalah.append(f"{duplikat_lintas_split} duplikat gambar lintas split")

    if masalah:
        tambah(
            hasil,
            "Manifest Kaggle",
            "TERBLOKIR",
            f"{len(baris)} baris; " + "; ".join(sorted(set(masalah))[:5]),
        )
    else:
        tambah(
            hasil,
            "Manifest Kaggle",
            "LULUS",
            f"{len(baris)} foto; train={jumlah_split['train']}, "
            f"val={jumlah_split['val']}, test={jumlah_split['test']}; "
            "semua path valid, isi gambar unik, dan grup tidak bocor antar-split.",
        )

    dataset_paths = (
        root / "data/public/garbage_v2/standardized_256",
        root / "dataset/standardized_256",
    )
    if any(path.is_dir() for path in dataset_paths):
        tambah(
            hasil,
            "Dataset Kaggle lokal",
            "LULUS",
            "Folder standardized_256 tersedia.",
        )
    else:
        tambah(
            hasil,
            "Dataset Kaggle lokal",
            "PERLU_TIM",
            "Dataset lokal tidak ditemukan; siapkan sumber dan lisensi sebelum regenerasi.",
        )
    return baris, jumlah_split


def periksa_model(
    root: Path, hasil: list[dict[str, str]], manifest: list[dict[str, str]]
) -> None:
    files = (
        root / "models/ecolens_mobilenetv2.pt",
        root / "models/ecolens_mobilenetv2.onnx",
        root / "models/face_detection_yunet_2023mar.onnx",
        root / "models/LICENSE-YUNET.txt",
        root / "models/labels.json",
        root / "reports/metrics_latih.json",
    )
    hilang = [path.relative_to(root).as_posix() for path in files if not path.is_file()]
    if hilang:
        tambah(hasil, "Artefak model", "TERBLOKIR", "Artefak hilang: " + ", ".join(hilang))
        return

    try:
        metrik = baca_json(root / "reports/metrics_latih.json")
        labels = baca_json(root / "models/labels.json")
    except (OSError, json.JSONDecodeError) as kesalahan:
        tambah(hasil, "Artefak model", "TERBLOKIR", f"Tidak dapat membaca JSON: {kesalahan}")
        return
    labels_urut = [labels.get(str(indeks)) for indeks in range(len(KELAS))]
    evaluasi = metrik.get("evaluasi_dataset_publik") or {}
    if labels_urut != list(KELAS) or not evaluasi.get("per_kelas"):
        tambah(hasil, "Evaluasi model", "TERBLOKIR", "Label atau metrik kelas tidak lengkap.")
        return

    latih_laporan = sum(metrik.get("jumlah_foto_latih_per_sumber", {}).values())
    latih_manifest = sum(
        item["split"] == "train" and item["sumber"] == "gd_v2" for item in manifest
    )
    uji_manifest = sum(
        item["split"] == "test" and item["sumber"] == "gd_v2" for item in manifest
    )
    uji_laporan = int(evaluasi.get("jumlah_foto_uji", 0))
    if latih_laporan == latih_manifest and uji_laporan == uji_manifest:
        tambah(
            hasil,
            "Evaluasi model",
            "LULUS",
            f"Seluruh split manifest tercakup; accuracy uji publik "
            f"{evaluasi['accuracy']:.1%}, macro F1 {evaluasi['macro_avg']['f1']:.1%}.",
        )
    else:
        tambah(
            hasil,
            "Evaluasi model",
            "PERLU_TIM",
            f"Laporan latih/uji {latih_laporan}/{uji_laporan}; "
            f"manifest latih/uji {latih_manifest}/{uji_manifest}.",
        )

    try:
        import onnx

        onnx.checker.check_model(str(root / "models/ecolens_mobilenetv2.onnx"))
    except (ImportError, OSError, ValueError) as kesalahan:
        tambah(hasil, "Model ONNX", "TERBLOKIR", f"Model ONNX tidak valid: {kesalahan}")
    else:
        tambah(hasil, "Model ONNX", "LULUS", "Model inferensi ONNX berhasil divalidasi.")


def periksa_analisis(root: Path, hasil: list[dict[str, str]]) -> None:
    path = root / "reports/analisis_kaggle.json"
    kandidat = root / "reports/kandidat_rekomendasi.md"
    if not path.is_file() or not kandidat.is_file():
        tambah(
            hasil,
            "Insight dan kandidat rekomendasi",
            "TERBLOKIR",
            "Jalankan python -m src.analyze_public untuk membuat laporan.",
        )
        return
    try:
        analisis = baca_json(path)
    except (OSError, json.JSONDecodeError) as kesalahan:
        tambah(hasil, "Insight dan kandidat rekomendasi", "TERBLOKIR", str(kesalahan))
        return
    jumlah_insight = len(analisis.get("insight_terhitung", []))
    jumlah_kandidat = len(analisis.get("kandidat_rekomendasi", []))
    if jumlah_insight >= 3 and jumlah_kandidat >= 2:
        tambah(
            hasil,
            "Insight dan kandidat rekomendasi",
            "LULUS",
            f"{jumlah_insight} insight terhitung dan {jumlah_kandidat} kandidat "
            "berbasis metrik publik; tim tetap menulis rekomendasi final.",
        )
    else:
        tambah(
            hasil,
            "Insight dan kandidat rekomendasi",
            "TERBLOKIR",
            f"Ditemukan {jumlah_insight} insight dan {jumlah_kandidat} kandidat; "
            "panduan meminta minimal tiga insight dan dua rekomendasi.",
        )


def periksa_dashboard(root: Path, hasil: list[dict[str, str]]) -> None:
    path = root / "app.py"
    if not path.is_file():
        tambah(hasil, "Dashboard lokal", "TERBLOKIR", "app.py tidak ditemukan.")
        return
    isi = path.read_text(encoding="utf-8")
    syarat = ("st.plotly_chart", "st.multiselect", "st.file_uploader", "onnxruntime")
    hilang = [item for item in syarat if item not in isi]
    if hilang:
        tambah(hasil, "Dashboard lokal", "TERBLOKIR", "Fitur belum ditemukan: " + ", ".join(hilang))
    else:
        tambah(
            hasil,
            "Dashboard lokal",
            "LULUS",
            "Visualisasi interaktif dan demo inferensi foto tersedia.",
        )


def periksa_kelengkapan_tim(root: Path, hasil: list[dict[str, str]]) -> None:
    deliverable = (
        ("Tautan dashboard publik tanpa login", root / "submission/dashboard_url.txt"),
        ("Ringkasan proyek PDF maksimal tiga halaman", root / "submission/ringkasan.pdf"),
        ("Tautan video demo", root / "submission/video_url.txt"),
        ("Surat pernyataan orisinalitas bertanda tangan", root / "submission/pernyataan_orisinalitas.pdf"),
    )
    for nama, path in deliverable:
        if path.is_file() and path.stat().st_size:
            rincian = "Berkas tersedia; tim perlu memeriksa akses dan isinya."
            if nama == "Ringkasan proyek PDF maksimal tiga halaman":
                rincian = (
                    "PDF tersedia; tim harus melengkapi placeholder nama dan memastikan "
                    "dokumen maksimal tiga halaman."
                )
            tambah(hasil, nama, "PERLU_TIM", rincian)
        else:
            tambah(hasil, nama, "PERLU_TIM", "Tim harus menyiapkan dan memverifikasi deliverable ini.")

    template = root / "content/rekomendasi.md"
    if template.is_file() and "[ISI TIM]" in template.read_text(encoding="utf-8"):
        tambah(
            hasil,
            "Rekomendasi final",
            "PERLU_TIM",
            "Tinjau kandidat berbasis metrik lalu tulis rekomendasi final tim.",
        )


def buat_laporan(root: Path) -> dict[str, object]:
    hasil: list[dict[str, str]] = []
    periksa_dokumen(root, hasil)
    manifest, _ = periksa_manifest(root, hasil)
    periksa_model(root, hasil, manifest)
    periksa_analisis(root, hasil)
    periksa_dashboard(root, hasil)
    periksa_kelengkapan_tim(root, hasil)

    status = (
        "BELUM SIAP"
        if any(item["status"] in {"TERBLOKIR", "PERLU_TIM"} for item in hasil)
        else "SIAP"
    )
    return {
        "status": status,
        "ruang_lingkup": "Demo klasifikasi sampah menggunakan dataset publik Kaggle",
        "pemeriksaan": hasil,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "checklist_pengumpulan.json")
    args = parser.parse_args()
    root = args.root.resolve()
    laporan = buat_laporan(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file_json:
        json.dump(laporan, file_json, indent=2, ensure_ascii=False)
    print(f"Status: {laporan['status']}")
    for item in laporan["pemeriksaan"]:
        print(f"[{item['status']}] {item['pemeriksaan']}: {item['rincian']}")
    print(f"Laporan tersimpan: {args.output}")
    return 0 if laporan["status"] == "SIAP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
