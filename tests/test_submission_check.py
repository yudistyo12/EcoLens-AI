import csv

from src.check_submission import buat_laporan
from src.check_submission import periksa_manifest


def test_pemeriksa_mengidentifikasi_deliverable_tim_yang_belum_ada(tmp_path):
    laporan = buat_laporan(tmp_path)

    assert laporan["status"] == "BELUM SIAP"
    assert laporan["ruang_lingkup"] == (
        "Demo klasifikasi sampah menggunakan dataset publik Kaggle"
    )
    pemeriksaan = {
        item["pemeriksaan"]: item["status"] for item in laporan["pemeriksaan"]
    }
    assert pemeriksaan["Tautan dashboard publik tanpa login"] == "PERLU_TIM"
    assert pemeriksaan["Ringkasan proyek PDF maksimal tiga halaman"] == "PERLU_TIM"
    assert "Data audit kampus" not in pemeriksaan


def tulis_manifest(root, isi_foto):
    folder_data = root / "data"
    folder_data.mkdir()
    baris = []
    for nomor, (isi, split) in enumerate(isi_foto):
        nama = f"foto_{nomor}.jpg"
        (folder_data / nama).write_bytes(isi)
        baris.append(
            {
                "path": f"data/{nama}",
                "kelas": "plastik",
                "sumber": "gd_v2",
                "group_id": f"grup_{nomor}",
                "split": split,
            }
        )
    path_manifest = root / "data/processed/manifest_latih.csv"
    path_manifest.parent.mkdir(parents=True, exist_ok=True)
    with path_manifest.open("w", newline="", encoding="utf-8") as file_csv:
        fieldnames = ("path", "kelas", "sumber", "group_id", "split")
        penulis = csv.DictWriter(file_csv, fieldnames=fieldnames)
        penulis.writeheader()
        penulis.writerows(baris)


def test_periksa_manifest_mendeteksi_duplikasi_gambar_antar_split(tmp_path):
    tulis_manifest(tmp_path, [(b"foto-sama", "train"), (b"foto-sama", "test")])
    hasil = []

    periksa_manifest(tmp_path, hasil)

    assert hasil[0]["status"] == "TERBLOKIR"
    assert "isi gambar duplikat" in hasil[0]["rincian"]
    assert "duplikat gambar lintas split" in hasil[0]["rincian"]


def test_periksa_manifest_menerima_gambar_unik_antar_split(tmp_path):
    tulis_manifest(tmp_path, [(b"foto-train", "train"), (b"foto-test", "test")])
    hasil = []

    periksa_manifest(tmp_path, hasil)

    assert hasil[0]["status"] == "LULUS"
    assert "isi gambar unik" in hasil[0]["rincian"]
