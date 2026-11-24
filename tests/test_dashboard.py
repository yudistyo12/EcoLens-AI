import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import src.photo_validation as photo_validation


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
ASSETS_DIR = APP_PATH.parent / "assets"


def radio_berlabel(app, label):
    return next(item for item in app.radio if item.label == label)


def test_dashboard_memuat_insight_dan_visualisasi_dataset_kaggle():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert any("EcoLens AI" in item.value for item in app.title)
    assert any("Apa yang dipelajari model?" in item.value for item in app.header)
    assert any("eco-sidebar-brand" in item.value for item in app.markdown)
    assert len(app.get("plotly_chart")) >= 2
    assert not any("DATA CONTOH (SINTETIK)" in item.value for item in app.error)
    teks_halaman = " ".join(
        item.value
        for tipe in ("title", "header", "subheader", "caption")
        for item in app.get(tipe)
        if hasattr(item, "value") and isinstance(item.value, str)
    )
    assert "demo" not in teks_halaman.lower()


def test_logo_dan_mark_merupakan_svg_valid():
    for nama in ("ecolens-logo.svg", "ecolens-mark.svg"):
        root = ET.parse(ASSETS_DIR / nama).getroot()
        assert root.tag.endswith("svg")


def test_setiap_halaman_dashboard_dapat_dibuka():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    halaman = (
        "Evaluasi model",
        "Coba klasifikasi",
        "Tindak lanjut",
        "Tentang model",
    )
    for nama in halaman:
        radio_berlabel(app, "Navigasi").set_value(nama).run()
        assert not app.exception

    radio_berlabel(app, "Navigasi").set_value("Evaluasi model").run()
    assert len(app.get("plotly_chart")) >= 3


def test_filter_metrik_kelas_boleh_dikosongkan():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    radio_berlabel(app, "Navigasi").set_value("Evaluasi model").run()
    app.multiselect[0].set_value([]).run()

    assert not app.exception
    assert any("Pilih setidaknya satu kelas" in item.value for item in app.info)


def test_klasifikasi_mendukung_kamera_dan_unggah_foto():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    radio_berlabel(app, "Navigasi").set_value("Coba klasifikasi").run()

    assert not app.exception
    assert len(app.get("camera_input")) == 1
    assert len(app.get("file_uploader")) == 0

    radio_berlabel(app, "Cara memasukkan foto").set_value("Unggah foto").run()

    assert not app.exception
    assert len(app.get("camera_input")) == 0
    assert len(app.get("file_uploader")) == 1


def test_foto_yang_terdeteksi_wajah_ditolak_sebelum_inferensi(monkeypatch):
    monkeypatch.setattr(photo_validation, "ada_wajah_terdeteksi", lambda gambar: True)
    isi_gambar = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(isi_gambar, format="PNG")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    radio_berlabel(app, "Navigasi").set_value("Coba klasifikasi").run()
    radio_berlabel(app, "Cara memasukkan foto").set_value("Unggah foto").run()
    app.get("file_uploader")[0].set_value(
        ("foto.png", isi_gambar.getvalue(), "image/png")
    ).run()

    assert not app.exception
    assert any("Wajah terdeteksi" in item.value for item in app.warning)
    assert not any(item.value == "Hasil prediksi" for item in app.subheader)


def test_hasil_prediksi_menampilkan_waktu_dan_rata_rata_sesi(monkeypatch):
    class SesiPalsu:
        def get_inputs(self):
            return [SimpleNamespace(name="gambar")]

        def run(self, outputs, inputs):
            assert outputs is None
            assert inputs["gambar"].shape == (1, 3, 224, 224)
            return [np.array([[10, 0, 0, 0, 0, 0]], dtype=np.float32)]

    def buat_sesi(*args, **kwargs):
        assert args
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        return SesiPalsu()

    monkeypatch.setattr(
        photo_validation, "ada_wajah_terdeteksi", lambda gambar: gambar.mode != "RGB"
    )
    monkeypatch.setattr(ort, "InferenceSession", buat_sesi)
    st.cache_resource.clear()

    isi_gambar = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(isi_gambar, format="PNG")

    try:
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        radio_berlabel(app, "Navigasi").set_value("Coba klasifikasi").run()
        radio_berlabel(app, "Cara memasukkan foto").set_value("Unggah foto").run()
        app.get("file_uploader")[0].set_value(
            ("foto.png", isi_gambar.getvalue(), "image/png")
        ).run()
        app.checkbox[0].set_value(True).run()

        assert not app.exception
        assert any(item.label == "Waktu inferensi model" for item in app.metric)
        assert len(app.session_state["waktu_inferensi_per_foto"]) == 1

        radio_berlabel(app, "Navigasi").set_value("Tentang model").run()

        assert any(item.label == "Ukuran model ONNX" for item in app.metric)
        assert any(
            item.label == "Rata-rata inferensi sesi (1 foto)" for item in app.metric
        )
    finally:
        st.cache_resource.clear()
