"""Membuat manifest foto latih dari Garbage V2 dan foto kampus."""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from src.config import (
    DATA_DIR,
    GD_V2_DIR,
    KELAS,
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLE_DIR,
    SEED,
    PEMETAAN_GD_V2,
    AMBANG_CONFIDENCE,
)
from src.utils import parse_nama_foto_latih


EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KOLOM_MANIFEST = ("path", "kelas", "sumber", "group_id", "split")
KOLOM_CONTOH = (
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


def cari_folder_dataset(folder_yang_diberikan: Path | None = None) -> Path:
    """Cari folder standardized_256 tanpa memakai varian dataset lain."""
    if folder_yang_diberikan is not None:
        folder = folder_yang_diberikan.resolve()
        if folder.name != "standardized_256":
            raise ValueError("Folder dataset harus bernama standardized_256.")
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
        return folder

    pilihan = (
        GD_V2_DIR,
        DATA_DIR / "dataset" / "standardized_256",
        DATA_DIR.parent / "dataset" / "standardized_256",
    )
    for folder in pilihan:
        if folder.is_dir():
            return folder.resolve()
    daftar = "\n".join(f"- {folder}" for folder in pilihan)
    raise FileNotFoundError(
        "Folder standardized_256 tidak ditemukan. Lokasi yang diperiksa:\n" + daftar
    )


def daftar_foto(folder: Path) -> list[Path]:
    """Ambil file gambar di dalam folder, termasuk subfoldernya."""
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in EKSTENSI_GAMBAR
        ),
        key=lambda path: str(path).lower(),
    )


def bagi_split(
    baris: list[dict[str, str]], seed: int = SEED
) -> list[dict[str, str]]:
    """Bagi data per kelas tanpa memisahkan foto dari grup objek yang sama."""
    grup_per_kelas: dict[str, list[str]] = {}
    for baris_data in baris:
        kelas = baris_data["kelas"]
        grup = baris_data["group_id"]
        grup_per_kelas.setdefault(kelas, [])
        if grup not in grup_per_kelas[kelas]:
            grup_per_kelas[kelas].append(grup)

    split_per_grup: dict[tuple[str, str], str] = {}
    for nomor_kelas, kelas in enumerate(sorted(grup_per_kelas)):
        grup = grup_per_kelas[kelas]
        random.Random(seed + nomor_kelas).shuffle(grup)
        jumlah = len(grup)

        if jumlah == 1:
            jumlah_latih, jumlah_validasi = 1, 0
        elif jumlah == 2:
            jumlah_latih, jumlah_validasi = 1, 1
        else:
            jumlah_latih = max(1, round(jumlah * 0.70))
            jumlah_validasi = max(1, round(jumlah * 0.15))
            if jumlah_latih + jumlah_validasi >= jumlah:
                jumlah_latih, jumlah_validasi = jumlah - 2, 1

        for indeks, id_grup in enumerate(grup):
            if indeks < jumlah_latih:
                nama_split = "train"
            elif indeks < jumlah_latih + jumlah_validasi:
                nama_split = "val"
            else:
                nama_split = "test"
            split_per_grup[(kelas, id_grup)] = nama_split

    return [
        {**baris_data, "split": split_per_grup[(baris_data["kelas"], baris_data["group_id"])]}
        for baris_data in baris
    ]


def ambil_foto_publik(folder_dataset: Path, max_publik: int) -> list[dict[str, str]]:
    """Tampilkan struktur dataset, petakan kelas, lalu sampling secara berulang."""
    folder_kelas = sorted(
        (path for path in folder_dataset.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    )
    foto_per_folder = {folder.name.lower(): daftar_foto(folder) for folder in folder_kelas}

    print(f"\nStruktur dataset yang dipakai: {folder_dataset}")
    print("Folder dan jumlah berkas gambar sebelum pemetaan:")
    for folder in folder_kelas:
        nama = folder.name.lower()
        jumlah = len(foto_per_folder[nama])
        tujuan = PEMETAAN_GD_V2.get(nama, "diabaikan")
        print(f"  {folder.name}: {jumlah} gambar -> {tujuan}")

    per_kelas: dict[str, list[tuple[str, Path]]] = {kelas: [] for kelas in KELAS}
    for nama_folder, kelas_tujuan in PEMETAAN_GD_V2.items():
        for foto in foto_per_folder.get(nama_folder, []):
            per_kelas[kelas_tujuan].append((nama_folder, foto))

    rng = random.Random(SEED)
    hasil = []
    for kelas in KELAS:
        pilihan = sorted(per_kelas[kelas], key=lambda item: str(item[1]).lower())
        rng.shuffle(pilihan)
        pilihan = pilihan[:max_publik]
        for nama_folder, foto in pilihan:
            path_rel = foto.relative_to(DATA_DIR.parent).as_posix()
            id_grup = f"gd_v2:{nama_folder}:{foto.relative_to(folder_dataset / nama_folder).as_posix()}"
            hasil.append(
                {
                    "path": path_rel,
                    "kelas": kelas,
                    "sumber": "gd_v2",
                    "group_id": id_grup,
                }
            )
    return hasil


def ambil_foto_kampus() -> tuple[list[dict[str, str]], int]:
    """Baca foto_latih; pola nama valid membagi foto dengan ID objek."""
    folder_latih = RAW_DIR / "foto_latih"
    if not folder_latih.is_dir():
        return [], 0

    hasil = []
    peringatan_pola = 0
    for kelas in KELAS:
        folder_kelas = folder_latih / kelas
        if not folder_kelas.is_dir():
            continue
        for foto in daftar_foto(folder_kelas):
            nama = parse_nama_foto_latih(foto.name)
            if nama is None or nama.kelas != kelas:
                peringatan_pola += 1
                id_grup = f"kampus:file:{foto.relative_to(folder_latih).as_posix()}"
            else:
                id_grup = f"kampus:{nama.group_id}"
            hasil.append(
                {
                    "path": foto.relative_to(DATA_DIR.parent).as_posix(),
                    "kelas": kelas,
                    "sumber": "kampus",
                    "group_id": id_grup,
                }
            )
    return hasil, peringatan_pola


def buat_csv_contoh(tujuan: Path) -> None:
    """Buat hanya file contoh dashboard yang secara eksplisit sintetik."""
    import datetime

    rng = random.Random(SEED)
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    with tujuan.open("w", newline="", encoding="utf-8") as file_csv:
        penulis = csv.DictWriter(file_csv, fieldnames=KOLOM_CONTOH)
        penulis.writeheader()
        for indeks in range(1, 31):
            menit = (indeks * 17) % (60 * 14)
            waktu_objek = datetime.datetime.strptime("05:00", "%H:%M") + datetime.timedelta(
                minutes=menit
            )
            jam = waktu_objek.strftime("%H:%M")
            if waktu_objek.hour < 11:
                periode = "pagi"
            elif waktu_objek.hour < 15:
                periode = "siang"
            else:
                periode = "sore"
            prediksi = rng.choice(KELAS)
            confidence = round(rng.uniform(0.35, 0.99), 3)
            penulis.writerow(
                {
                    "photo_id": f"CONTOH_{indeks:03d}",
                    "file_name": f"CONTOH_{indeks:03d}.jpg",
                    "kode_lokasi": f"L{(indeks - 1) % 10 + 1:02d}",
                    "tanggal": f"2026-01-{(indeks - 1) % 28 + 1:02d}",
                    "jam": jam,
                    "waktu": periode,
                    "label_manual": "",
                    "prediksi_model": prediksi,
                    "confidence": confidence,
                    "perlu_verifikasi": confidence < AMBANG_CONFIDENCE,
                    "sumber": "CONTOH_SINTETIK",
                }
            )


def cetak_distribusi(baris: list[dict[str, str]]) -> None:
    """Cetak jumlah foto per kelas dan sumber dari manifest yang dibuat."""
    jumlah = Counter((item["kelas"], item["sumber"]) for item in baris)
    print("\nJumlah foto per kelas dan sumber (manifest aktual):")
    for kelas in KELAS:
        print(
            f"  {kelas}: "
            + ", ".join(
                f"{sumber}={jumlah.get((kelas, sumber), 0)}"
                for sumber in ("gd_v2", "trashnet", "kampus")
            )
        )

    for kelas in KELAS:
        if jumlah.get((kelas, "kampus"), 0) < 60:
            print(
                f"PERINGATAN: foto latih kampus kelas {kelas} kurang dari 60 "
                f"({jumlah.get((kelas, 'kampus'), 0)})."
            )


def jalankan(folder_dataset: Path | None, max_publik: int) -> Path:
    if max_publik < 1:
        raise ValueError("--max-publik harus minimal 1.")

    folder = cari_folder_dataset(folder_dataset)
    baris = ambil_foto_publik(folder, max_publik)
    foto_kampus, jumlah_nama_invalid = ambil_foto_kampus()
    baris.extend(foto_kampus)
    baris = bagi_split(baris)

    if jumlah_nama_invalid:
        print(
            "PERINGATAN: "
            f"{jumlah_nama_invalid} nama foto kampus tidak sesuai pola; "
            "masing-masing diperlakukan sebagai grup sendiri dan mungkin bocor."
        )
    if not any(item["sumber"] == "trashnet" for item in baris):
        print("TrashNet tidak digunakan; tidak ada unduhan dataset tambahan.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = PROCESSED_DIR / "manifest_latih.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file_csv:
        penulis = csv.DictWriter(file_csv, fieldnames=KOLOM_MANIFEST)
        penulis.writeheader()
        penulis.writerows(baris)

    cetak_distribusi(baris)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    buat_csv_contoh(SAMPLE_DIR / "hasil_prediksi_CONTOH.csv")
    print(f"\nManifest tersimpan: {manifest}")
    print(f"Data contoh sintetik tersimpan: {SAMPLE_DIR / 'hasil_prediksi_CONTOH.csv'}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Folder standardized_256; jika kosong, cari lokasi dataset yang dikenal.",
    )
    parser.add_argument(
        "--max-publik",
        type=int,
        default=300,
        help="Batas foto publik per kelas hasil pemetaan (default: 300).",
    )
    args = parser.parse_args()

    try:
        jalankan(args.dataset_dir, args.max_publik)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
