import pytest

from src.analyze import baca_hasil
from src.predict import cari_foto_audit


def test_prediksi_menolak_folder_foto_audit_yang_belum_ada(tmp_path):
    with pytest.raises(FileNotFoundError, match="Folder foto audit tidak ditemukan"):
        cari_foto_audit(tmp_path / "foto_audit")


def test_analisis_menolak_csv_hasil_yang_belum_ada(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hasil prediksi belum tersedia"):
        baca_hasil(tmp_path / "hasil_prediksi.csv")
