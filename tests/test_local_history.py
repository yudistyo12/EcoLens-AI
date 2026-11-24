from PIL import Image

from src.local_history import (
    baca_riwayat_lokal,
    hapus_riwayat_lokal,
    penyimpanan_lokal_diizinkan,
    simpan_hasil_lokal,
)


def test_riwayat_lokal_menyimpan_hasil_dan_foto_sanitasi(tmp_path):
    gambar = Image.new("RGB", (224, 224), color=(30, 120, 60))
    top3 = [{"Kelas": "plastik", "Skor softmax": 0.9}]

    tersimpan = simpan_hasil_lokal(
        gambar,
        "plastik",
        0.9,
        top3,
        direktori=tmp_path,
    )

    riwayat = baca_riwayat_lokal(direktori=tmp_path)
    assert len(riwayat) == 1
    assert riwayat[0]["kelas_prediksi"] == "plastik"
    foto_tersimpan = Image.open(tmp_path / tersimpan["nama_foto"])
    assert foto_tersimpan.format == "JPEG"
    assert foto_tersimpan.size == (224, 224)


def test_riwayat_lokal_dapat_dihapus_dengan_batas_direktori(tmp_path):
    hasil = simpan_hasil_lokal(
        Image.new("RGB", (20, 20)),
        "kaca",
        0.75,
        [],
        direktori=tmp_path,
    )
    path_foto = tmp_path / hasil["nama_foto"]

    hapus_riwayat_lokal(direktori=tmp_path)

    assert not path_foto.exists()
    assert baca_riwayat_lokal(direktori=tmp_path) == []


def test_server_riwayat_hanya_diizinkan_di_localhost():
    assert penyimpanan_lokal_diizinkan("http://localhost:8501/")
    assert penyimpanan_lokal_diizinkan("http://127.0.0.1:8501/")
    assert not penyimpanan_lokal_diizinkan("https://ecolens.example.com/")
