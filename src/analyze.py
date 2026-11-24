"""Hitung ringkasan pola foto audit kampus dan kandidat rekomendasi."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.config import AMBANG_CONFIDENCE, KELAS, PROCESSED_DIR, REPORT_DIR
from src.utils import tentukan_waktu


KELAS_POTENSI_PILAH = ("plastik", "kertas_kardus", "logam", "kaca")
KOLOM_WAJIB = {
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
}


def baca_hasil(path_hasil: Path) -> list[dict[str, str]]:
    """Baca hasil prediksi kampus dan tolak data contoh/sumber non-kampus."""
    if not path_hasil.is_file():
        raise FileNotFoundError(
            f"Hasil prediksi belum tersedia: {path_hasil}. "
            "Jalankan src.predict setelah foto audit dan model siap."
        )

    with path_hasil.open("r", newline="", encoding="utf-8-sig") as file_csv:
        pembaca = csv.DictReader(file_csv)
        if not KOLOM_WAJIB.issubset(pembaca.fieldnames or []):
            raise ValueError(f"CSV hasil harus memiliki kolom: {sorted(KOLOM_WAJIB)}")
        baris = list(pembaca)

    if not baris:
        raise ValueError("CSV hasil prediksi tidak memiliki baris data.")
    if any(item["sumber"] == "CONTOH_SINTETIK" for item in baris):
        raise ValueError(
            "Analisis ditolak: hasil_prediksi.csv berisi data contoh sintetik."
        )
    if any(item["sumber"] != "kampus" for item in baris):
        raise ValueError("Analisis lokasi hanya menerima baris dengan sumber=kampus.")

    id_foto = set()
    for item in baris:
        if not item["photo_id"] or item["photo_id"] in id_foto:
            raise ValueError("photo_id kosong atau duplikat pada hasil prediksi.")
        id_foto.add(item["photo_id"])
        if item["prediksi_model"] not in KELAS:
            raise ValueError(f"Prediksi kelas tidak dikenal: {item['prediksi_model']}")
        try:
            datetime.strptime(item["tanggal"], "%Y-%m-%d")
            waktu_objek = datetime.strptime(item["jam"], "%H:%M")
            confidence = float(item["confidence"])
        except ValueError as kesalahan:
            raise ValueError(
                f"Tanggal, jam, atau confidence tidak valid untuk {item['photo_id']}."
            ) from kesalahan
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence di luar rentang 0-1: {item['photo_id']}")
        if item["waktu"] != tentukan_waktu(waktu_objek.strftime("%H:%M")):
            raise ValueError(f"Kategori waktu tidak cocok dengan jam: {item['photo_id']}")
        if item["perlu_verifikasi"].lower() not in {"true", "false"}:
            raise ValueError(f"Nilai perlu_verifikasi tidak valid: {item['photo_id']}")
    return baris


def hitung_komposisi(
    baris: list[dict[str, str]], kunci_grup: str
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Hitung jumlah foto dan persentase kelas dalam setiap kelompok."""
    hitungan: dict[str, Counter] = defaultdict(Counter)
    for item in baris:
        hitungan[item[kunci_grup]][item["prediksi_model"]] += 1

    hasil = []
    jumlah_per_grup = {}
    for grup in sorted(hitungan):
        total = sum(hitungan[grup].values())
        jumlah_per_grup[grup] = total
        for kelas in KELAS:
            jumlah = hitungan[grup][kelas]
            hasil.append(
                {
                    kunci_grup: grup,
                    "kelas": kelas,
                    "jumlah_foto": jumlah,
                    "persentase": round(jumlah / total * 100, 2) if total else 0.0,
                }
            )
    return hasil, jumlah_per_grup


def tulis_csv(path: Path, baris: list[dict[str, object]], kolom: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_csv:
        penulis = csv.DictWriter(file_csv, fieldnames=kolom)
        penulis.writeheader()
        penulis.writerows(baris)


def nilai_tertinggi(jumlah: dict[str, int]) -> list[str]:
    if not jumlah:
        return []
    nilai_maksimum = max(jumlah.values())
    return sorted(kunci for kunci, nilai in jumlah.items() if nilai == nilai_maksimum)


def buat_kandidat(
    baris: list[dict[str, str]],
    jumlah_lokasi: dict[str, int],
    jumlah_waktu: dict[str, int],
    komposisi_lokasi: list[dict[str, object]],
    total_per_kelas: Counter,
    jumlah_verifikasi: int,
) -> str:
    """Susun kandidat berbasis hitungan aktual; tim tetap menentukan rekomendasi."""
    jumlah_foto = len(baris)
    lokasi_puncak = nilai_tertinggi(jumlah_lokasi)
    waktu_puncak = nilai_tertinggi(jumlah_waktu)
    baris_lokasi_puncak = [
        item
        for item in komposisi_lokasi
        if item["kode_lokasi"] in lokasi_puncak and item["jumlah_foto"] > 0
    ]

    hasil = [
        "# Kandidat Rekomendasi (perlu ditinjau tim)",
        "",
        "Dokumen ini dibuat otomatis dari hasil foto audit kampus. Ini bukan",
        "rekomendasi final; tim harus meninjau konteks lapangan sebelum mengusulkannya.",
        "Satuan yang dihitung adalah jumlah foto/item, bukan berat atau volume.",
        "Komposisi memakai prediksi model; foto berlabel manual digunakan terpisah",
        "untuk evaluasi model.",
        "",
    ]
    if lokasi_puncak and baris_lokasi_puncak:
        kelas_lokasi_puncak = max(
            baris_lokasi_puncak,
            key=lambda item: (int(item["jumlah_foto"]), str(item["kelas"])),
        )
        hasil.extend(
            [
                "## Kandidat: tinjau fasilitas pemilahan di lokasi dengan foto terbanyak",
                "",
                (
                    f"Lokasi {lokasi_puncak[0]} memiliki jumlah foto tertinggi "
                    f"({jumlah_lokasi[lokasi_puncak[0]]} dari {jumlah_foto}). "
                    f"Pada lokasi tersebut, kelas prediksi terbanyak adalah "
                    f"{kelas_lokasi_puncak['kelas']} "
                    f"({kelas_lokasi_puncak['jumlah_foto']} foto; "
                    f"{kelas_lokasi_puncak['persentase']}%)."
                ),
                "Tinjau kondisi tempat sampah dan kecocokan pemilahan di lokasi ini.",
                "",
            ]
        )

    if waktu_puncak:
        hasil.extend(
            [
                "## Kandidat: tinjau jadwal pengangkutan pada periode tersibuk",
                "",
                (
                    f"Periode dengan jumlah foto tertinggi: {', '.join(waktu_puncak)} "
                    f"({jumlah_waktu[waktu_puncak[0]]} foto dari {jumlah_foto})."
                ),
                "Cocokkan pola ini dengan catatan operasional sebelum mengubah jadwal.",
                "",
            ]
        )

    jumlah_pilah = sum(total_per_kelas[kelas] for kelas in KELAS_POTENSI_PILAH)
    hasil.extend(
        [
            "## Ringkasan kategori berpotensi dipilah",
            "",
            (
                f"Kelas plastik, kertas_kardus, logam, dan kaca mencakup "
                f"{jumlah_pilah} dari {jumlah_foto} foto "
                f"({jumlah_pilah / jumlah_foto * 100:.2f}%)."
            ),
            "Angka ini bukan pengukuran berat, volume, atau bukti bahwa barang dapat didaur ulang.",
            "",
            "## Pemeriksaan sebelum menjadi rekomendasi final",
            "",
            f"- Foto yang perlu verifikasi model: {jumlah_verifikasi} dari {jumlah_foto}.",
            "- Periksa foto yang confidence-nya rendah dan kecocokan kelas di lapangan.",
            "- Catat keputusan final hanya setelah ditinjau dan disetujui tim.",
        ]
    )
    return "\n".join(hasil) + "\n"


def jalankan(path_input: Path) -> dict[str, Path]:
    baris = baca_hasil(path_input)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    per_lokasi, jumlah_lokasi = hitung_komposisi(baris, "kode_lokasi")
    per_waktu, jumlah_waktu = hitung_komposisi(baris, "waktu")
    per_hari, jumlah_hari = hitung_komposisi(baris, "tanggal")
    total_per_kelas = Counter(item["prediksi_model"] for item in baris)
    jumlah_verifikasi = sum(
        item["perlu_verifikasi"].lower() == "true" for item in baris
    )
    jumlah_pilah = sum(total_per_kelas[kelas] for kelas in KELAS_POTENSI_PILAH)

    lokasi_puncak = nilai_tertinggi(jumlah_lokasi)
    waktu_puncak = nilai_tertinggi(jumlah_waktu)
    hari_puncak = nilai_tertinggi(jumlah_hari)
    ringkasan = {
        "jumlah_foto": len(baris),
        "satuan": "foto/item, bukan berat atau volume",
        "jumlah_per_kelas": {kelas: total_per_kelas[kelas] for kelas in KELAS},
        "lokasi_puncak": [
            {"kode_lokasi": lokasi, "jumlah_foto": jumlah_lokasi[lokasi]}
            for lokasi in lokasi_puncak
        ],
        "waktu_puncak": [
            {"waktu": waktu, "jumlah_foto": jumlah_waktu[waktu]}
            for waktu in waktu_puncak
        ],
        "hari_puncak": [
            {"tanggal": tanggal, "jumlah_foto": jumlah_hari[tanggal]}
            for tanggal in hari_puncak
        ],
        "jumlah_perlu_verifikasi": jumlah_verifikasi,
        "persentase_perlu_verifikasi": round(jumlah_verifikasi / len(baris) * 100, 2),
        "potensi_kategori_pilah": {
            "kelas": list(KELAS_POTENSI_PILAH),
            "jumlah_foto": jumlah_pilah,
            "persentase": round(jumlah_pilah / len(baris) * 100, 2),
            "catatan": (
                "Hitungan berbasis kelas prediksi, bukan berat/volume atau kepastian "
                "bahan dapat didaur ulang."
            ),
        },
        "ambang_confidence_perlu_verifikasi": AMBANG_CONFIDENCE,
        "catatan": (
            "Analisis menggunakan prediksi_model; label_manual hanya untuk evaluasi. "
            "Insight menggambarkan foto yang dikumpulkan, bukan otomatis seluruh kampus."
        ),
    }

    path_lokasi = REPORT_DIR / "komposisi_lokasi.csv"
    path_waktu = REPORT_DIR / "komposisi_waktu.csv"
    path_hari = REPORT_DIR / "komposisi_hari.csv"
    path_ringkasan = REPORT_DIR / "analisis_audit.json"
    path_kandidat = REPORT_DIR / "kandidat_rekomendasi.md"
    tulis_csv(
        path_lokasi,
        per_lokasi,
        ["kode_lokasi", "kelas", "jumlah_foto", "persentase"],
    )
    tulis_csv(path_waktu, per_waktu, ["waktu", "kelas", "jumlah_foto", "persentase"])
    tulis_csv(path_hari, per_hari, ["tanggal", "kelas", "jumlah_foto", "persentase"])
    with path_ringkasan.open("w", encoding="utf-8") as file_json:
        json.dump(ringkasan, file_json, indent=2, ensure_ascii=False)

    with path_kandidat.open("w", encoding="utf-8") as file_md:
        file_md.write(
            buat_kandidat(
                baris,
                jumlah_lokasi,
                jumlah_waktu,
                per_lokasi,
                total_per_kelas,
                jumlah_verifikasi,
            )
        )

    print(f"Jumlah foto kampus dianalisis: {len(baris)}")
    print(f"Foto perlu verifikasi: {jumlah_verifikasi}")
    print(f"Jumlah kandidat kategori pilah: {jumlah_pilah}")
    print(f"Ringkasan tersimpan: {path_ringkasan}")
    print(f"Kandidat (perlu ditinjau tim): {path_kandidat}")
    return {
        "lokasi": path_lokasi,
        "waktu": path_waktu,
        "hari": path_hari,
        "ringkasan": path_ringkasan,
        "kandidat": path_kandidat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "hasil_prediksi.csv",
        help="CSV hasil prediksi foto audit kampus.",
    )
    args = parser.parse_args()
    try:
        jalankan(args.input)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
