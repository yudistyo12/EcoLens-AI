import pytest
from PIL import Image

from src.photo_validation import MODEL_WAJAH, ada_wajah_terdeteksi


def test_detektor_wajah_tidak_mendeteksi_gambar_kosong():
    assert not ada_wajah_terdeteksi(Image.new("RGB", (320, 320), color="white"))


def test_detektor_wajah_gagal_jelas_jika_model_tidak_ada(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model deteksi wajah tidak ditemukan"):
        ada_wajah_terdeteksi(
            Image.new("RGB", (320, 320)), tmp_path / "model-tidak-ada.onnx"
        )


def test_model_deteksi_wajah_tersedia():
    assert MODEL_WAJAH.is_file()
