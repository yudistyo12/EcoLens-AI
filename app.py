"""Dashboard EcoLens AI untuk klasifikasi sampah berbasis dataset Kaggle."""

import hashlib
import json
import sqlite3
import time
from html import escape
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.config import AMBANG_CONFIDENCE, KELAS, MODEL_DIR, PROJECT_ROOT, REPORT_DIR
from src.local_history import (
    LOCAL_HISTORY_DIR,
    baca_riwayat_lokal,
    hapus_riwayat_lokal,
    penyimpanan_lokal_diizinkan,
    simpan_hasil_lokal,
)
from src.photo_validation import ada_wajah_terdeteksi


MEAN_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)
UKURAN_INPUT = 224
WARNA_KELAS = {
    "plastik": "#4C8C6B",
    "kertas_kardus": "#B89A52",
    "logam": "#617A8B",
    "kaca": "#4D94A0",
    "organik": "#7A9651",
    "residu": "#8D7772",
}

GAYA = """
<style>
    :root {
        --eco-ink: #1d2b24;
        --eco-muted: #68756d;
        --eco-green: #285d46;
        --eco-forest: #193d30;
        --eco-lime: #dce8a8;
        --eco-line: #dfe6de;
        --eco-paper: #f5f6f2;
        --eco-white: #ffffff;
        --eco-shadow: 0 10px 28px rgba(29, 43, 36, 0.06);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 92% 0%, rgba(220, 232, 168, 0.18), transparent 27rem),
            var(--eco-paper);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMain"] .block-container {
        max-width: 1360px;
        padding: 2.25rem 3.1rem 3.5rem;
    }
    h1, h2, h3 {
        color: var(--eco-ink);
        letter-spacing: -0.035em;
    }
    h1 {
        font-size: 2.55rem !important;
        line-height: 1.04 !important;
        margin: 0.55rem 0 0.2rem !important;
    }
    h2 {
        font-size: 1.62rem !important;
        margin-top: 1.35rem !important;
    }
    h3 {
        font-size: 1.12rem !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: var(--eco-muted);
    }
    [data-testid="stSidebar"] {
        background: var(--eco-forest);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #d7e4dc;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #f5f8f4 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 0.78rem;
        padding: 0.52rem 0.72rem;
        transition: background 140ms ease, border-color 140ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
        outline: 2px solid var(--eco-lime);
        outline-offset: 2px;
    }
    [data-testid="stMetric"] {
        background: var(--eco-white);
        border: 1px solid var(--eco-line);
        border-radius: 1rem;
        padding: 1.05rem 1.2rem;
        min-height: 116px;
        box-shadow: var(--eco-shadow);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--eco-muted);
        font-size: 0.82rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--eco-ink);
        font-weight: 700;
        letter-spacing: -0.035em;
    }
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        background: var(--eco-white);
        border: 1px solid var(--eco-line);
        border-radius: 1rem;
        padding: 0.4rem;
        box-shadow: 0 4px 16px rgba(29, 43, 36, 0.025);
    }
    [data-testid="stAlert"] {
        border-radius: 0.85rem;
    }
    [data-testid="stFileUploader"] section {
        background: var(--eco-white);
        border: 1px dashed #9aab9d;
        border-radius: 1rem;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 2.75rem;
        border-radius: 0.78rem;
        font-weight: 650;
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(29, 43, 36, 0.1);
    }
    div.stButton > button:focus-visible,
    div.stDownloadButton > button:focus-visible {
        outline: 3px solid rgba(40, 93, 70, 0.28);
        outline-offset: 2px;
    }
    .eco-brandline {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--eco-green);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.105em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .eco-brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.15rem;
        height: 2.15rem;
        flex: 0 0 auto;
    }
    .eco-brand-mark svg {
        display: block;
        width: 100%;
        height: 100%;
    }
    .eco-pill {
        display: inline-block;
        border: 1px solid #d6e0d6;
        border-radius: 999px;
        color: #526358;
        background: rgba(255, 255, 255, 0.8);
        padding: 0.36rem 0.7rem;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .eco-section-label {
        color: var(--eco-green);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.55rem 0 0.25rem;
    }
    .eco-insight {
        height: 100%;
        min-height: 132px;
        background: white;
        border: 1px solid var(--eco-line);
        border-left: 3px solid var(--eco-green);
        border-radius: 0.85rem;
        padding: 1rem 1.05rem;
        box-shadow: 0 4px 15px rgba(29, 43, 36, 0.03);
    }
    .eco-insight-number {
        color: var(--eco-green);
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.11em;
    }
    .eco-insight-text {
        color: var(--eco-ink);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-top: 0.55rem;
    }
    .eco-sidebar-brand {
        display: block;
        margin: 0.3rem 0 1.5rem;
        padding: 0.45rem 0.55rem;
        border-radius: 1rem;
        background: #f2f4eb;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.1);
    }
    .eco-sidebar-brand svg {
        display: block;
        width: 100%;
        height: auto;
    }
    .eco-result-label {
        color: var(--eco-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .eco-result-value {
        color: var(--eco-green);
        font-size: 1.9rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin: 0.1rem 0 0.3rem;
    }
    .eco-footer {
        border-top: 1px solid var(--eco-line);
        color: var(--eco-muted);
        font-size: 0.76rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
    }
    @media (max-width: 800px) {
        [data-testid="stMain"] .block-container {
            padding: 1.5rem 1.1rem 2.5rem;
        }
        h1 {
            font-size: 2.1rem !important;
        }
        .eco-brandline {
            flex-wrap: wrap;
            letter-spacing: 0.07em;
        }
    }
</style>
"""


def baca_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Laporan belum tersedia: {path}")
    with path.open("r", encoding="utf-8") as file_json:
        return json.load(file_json)


def muat_data() -> tuple[dict, dict, pd.DataFrame]:
    analisis = baca_json(REPORT_DIR / "analisis_kaggle.json")
    metrik = baca_json(REPORT_DIR / "metrics_latih.json")
    path_manifest = PROJECT_ROOT / "data/processed/manifest_latih.csv"
    if not path_manifest.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {path_manifest}")

    manifest = pd.read_csv(path_manifest)
    kolom_wajib = {"kelas", "sumber", "split"}
    if not kolom_wajib.issubset(manifest.columns):
        raise ValueError("Manifest tidak memiliki kolom kelas, sumber, dan split.")
    if not manifest["kelas"].isin(KELAS).all():
        raise ValueError("Manifest berisi kelas yang tidak dikenal.")
    if not manifest["split"].isin({"train", "val", "test"}).all():
        raise ValueError("Manifest berisi nama split yang tidak dikenal.")
    manifest = manifest[manifest["sumber"] == "gd_v2"].copy()
    if manifest.empty:
        raise ValueError("Manifest tidak berisi data sumber GD V2.")
    return analisis, metrik, manifest


def hitung_cakupan_metrik(metrik: dict, manifest: pd.DataFrame) -> bool:
    jumlah_latih = sum(metrik.get("jumlah_foto_latih_per_sumber", {}).values())
    jumlah_latih_manifest = int((manifest["split"] == "train").sum())
    evaluasi = metrik.get("evaluasi_dataset_publik") or {}
    jumlah_uji = int(evaluasi.get("jumlah_foto_uji", 0))
    jumlah_uji_manifest = int((manifest["split"] == "test").sum())
    return (
        jumlah_latih == jumlah_latih_manifest
        and jumlah_uji == jumlah_uji_manifest
    )


def siapkan_gambar(file_gambar) -> tuple[Image.Image, np.ndarray, Image.Image]:
    try:
        with Image.open(file_gambar) as gambar:
            gambar.getexif().clear()
            gambar_asli = gambar.convert("RGB")
    except (OSError, UnidentifiedImageError) as kesalahan:
        raise ValueError("File yang diunggah bukan gambar yang dapat dibaca.") from kesalahan

    ukuran = getattr(Image, "Resampling", Image)
    gambar_rgb = gambar_asli.resize((UKURAN_INPUT, UKURAN_INPUT), ukuran.BILINEAR)
    array = np.asarray(gambar_rgb, dtype=np.float32) / 255.0
    array = (array - MEAN_IMAGENET) / STD_IMAGENET
    tensor = np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)
    return gambar_rgb, tensor, gambar_asli


@st.cache_resource
def muat_sesi_onnx(path_model: str, waktu_ubah: float):
    del waktu_ubah
    path = Path(path_model)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model ONNX tidak ditemukan: {path}. Jalankan python -m src.export_onnx."
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def simpan_lokal_aktif() -> bool:
    alamat = getattr(st.context, "url", "") or ""
    return penyimpanan_lokal_diizinkan(alamat)


def gaya_grafik(grafik):
    grafik.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#394a40", "size": 12},
        margin={"l": 12, "r": 18, "t": 58, "b": 12},
        title={"font": {"size": 16, "color": "#1d2b24"}, "x": 0.02, "xanchor": "left"},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    grafik.update_xaxes(showline=False, zeroline=False, gridcolor="#e8ece7")
    grafik.update_yaxes(showline=False, zeroline=False, gridcolor="#e8ece7")
    return grafik


def label_bagian(teks: str) -> None:
    st.markdown(f'<div class="eco-section-label">{escape(teks)}</div>', unsafe_allow_html=True)


def tampilkan_ikhtisar(analisis: dict, metrik: dict, manifest: pd.DataFrame) -> None:
    label_bagian("01 / Gambaran data")
    st.header("Apa yang dipelajari model?")

    data_asal = analisis["dataset_asal"]
    evaluasi = metrik["evaluasi_dataset_publik"]
    kolom = st.columns(3)
    kolom[0].metric("Foto uji", f"{evaluasi['jumlah_foto_uji']:,}")
    kolom[1].metric("Accuracy", f"{evaluasi['accuracy']:.1%}")
    kolom[2].metric("Macro F1", f"{evaluasi['macro_avg']['f1']:.1%}")

    st.markdown('<div class="eco-section-label">TEMUAN YANG DIHITUNG</div>', unsafe_allow_html=True)
    insight_kolom = st.columns(len(analisis["insight_terhitung"]))
    for nomor, (kolom_insight, item) in enumerate(
        zip(insight_kolom, analisis["insight_terhitung"]), start=1
    ):
        kolom_insight.markdown(
            f'<div class="eco-insight">'
            f'<div class="eco-insight-number">TEMUAN {nomor:02}</div>'
            f'<div class="eco-insight-text">{escape(item)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    sumber = pd.DataFrame(
        [
            {
                "Folder": folder,
                "Pemetaan kelas": data_asal.get("pemetaan_per_folder", {}).get(
                    folder, "diabaikan"
                ),
                "Jumlah foto": jumlah,
                "Status": (
                    "Dipakai"
                    if data_asal.get("pemetaan_per_folder", {}).get(
                        folder, "diabaikan"
                    )
                    != "diabaikan"
                    else "Diabaikan"
                ),
            }
            for folder, jumlah in data_asal["jumlah_foto_per_folder"].items()
        ]
    )
    sumber["Jumlah foto"] = pd.to_numeric(sumber["Jumlah foto"])
    sumber = sumber.sort_values("Jumlah foto", ascending=True)
    chart_sumber = px.bar(
        sumber,
        x="Jumlah foto",
        y="Folder",
        color="Status",
        orientation="h",
        title="Jumlah file yang ditemukan per folder sumber",
        color_discrete_map={"Dipakai": "#397653", "Diabaikan": "#b9c2ba"},
        hover_data={"Pemetaan kelas": True, "Status": False},
    )
    chart_sumber.update_layout(height=390, showlegend=True)
    st.plotly_chart(gaya_grafik(chart_sumber), width="stretch")
    with st.expander("Lihat tabel pemetaan folder"):
        st.dataframe(
            sumber.sort_values("Jumlah foto", ascending=False),
            hide_index=True,
            width="stretch",
        )

    label_bagian("DATA YANG MASUK KE MODEL")
    st.subheader("Pembagian sampel")
    kiri, kanan = st.columns([1, 2])
    with kiri:
        kelas_pilih = st.multiselect(
            "Kelas",
            options=list(KELAS),
            default=list(KELAS),
            key="filter_kelas_dataset",
        )
        split_pilih = st.multiselect(
            "Split",
            options=("train", "val", "test"),
            default=("train", "val", "test"),
            key="filter_split_dataset",
        )
        terfilter = manifest[
            manifest["kelas"].isin(kelas_pilih)
            & manifest["split"].isin(split_pilih)
        ]
        st.metric("Sampel terpilih", f"{len(terfilter):,} / {len(manifest):,}")
    with kanan:
        if not terfilter.empty:
            per_split = (
                terfilter.groupby(["kelas", "split"])
                .size()
                .rename("Jumlah foto")
                .reset_index()
            )
            grafik_split = px.bar(
                per_split,
                x="kelas",
                y="Jumlah foto",
                color="split",
                barmode="group",
                category_orders={"kelas": list(KELAS)},
                color_discrete_map={
                    "train": "#245b43",
                    "val": "#8da98d",
                    "test": "#d2aa57",
                },
                title="Foto manifest menurut kelas dan split",
                labels={"kelas": "Kelas"},
            )
            grafik_split.update_layout(height=370)
            st.plotly_chart(gaya_grafik(grafik_split), width="stretch")
        else:
            st.info("Pilih setidaknya satu kelas dan satu split.")

def tampilkan_evaluasi(metrik: dict, manifest: pd.DataFrame) -> None:
    label_bagian("02 / Kinerja model")
    st.header("Seberapa baik model mengenali kelas?")
    evaluasi = metrik["evaluasi_dataset_publik"]
    if not hitung_cakupan_metrik(metrik, manifest):
        st.warning(
            "Metrik menggunakan subset manifest. Periksa kembali jumlah foto "
            "sebelum mengutip hasil."
        )

    ringkas = st.columns(3)
    ringkas[0].metric("Akurasi", f"{evaluasi['accuracy']:.1%}")
    ringkas[1].metric("F1 makro", f"{evaluasi['macro_avg']['f1']:.1%}")
    ringkas[2].metric("Foto uji Kaggle", f"{evaluasi['jumlah_foto_uji']:,}")

    tabel_kelas = (
        pd.DataFrame(evaluasi["per_kelas"])
        .T.rename_axis("Kelas")
        .reindex(list(KELAS))
        .reset_index()
        .rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "support": "Jumlah foto uji",
            }
        )
    )
    kelas_pilih = st.multiselect(
        "Pilih kelas untuk dibandingkan",
        options=list(KELAS),
        default=list(KELAS),
        key="filter_kelas_metrik",
    )
    if not kelas_pilih:
        st.info("Pilih setidaknya satu kelas untuk menampilkan evaluasi.")
        return
    tabel_pilih = tabel_kelas[tabel_kelas["Kelas"].isin(kelas_pilih)]

    f1_kiri, f1_kanan = st.columns([1.35, 1])
    with f1_kiri:
        grafik_f1 = px.bar(
            tabel_pilih.sort_values("F1"),
            x="F1",
            y="Kelas",
            color="Kelas",
            orientation="h",
            range_x=[0, 1],
            color_discrete_map=WARNA_KELAS,
            title="F1 per kelas",
            labels={"F1": "Skor F1"},
            hover_data={"Precision": True, "Recall": True},
        )
        grafik_f1.update_layout(showlegend=False, height=370)
        grafik_f1.update_xaxes(tickformat=".0%")
        st.plotly_chart(gaya_grafik(grafik_f1), width="stretch")
    with f1_kanan:
        matriks = np.asarray(evaluasi["confusion_matrix"])
        indeks = [KELAS.index(kelas) for kelas in kelas_pilih]
        matriks_pilih = matriks[np.ix_(indeks, indeks)]
        grafik_matriks = px.imshow(
            matriks_pilih,
            x=kelas_pilih,
            y=kelas_pilih,
            labels={"x": "Prediksi", "y": "Label sebenarnya", "color": "Foto"},
            text_auto=True,
            title="Confusion matrix",
            color_continuous_scale=["#f1f5f1", "#245b43"],
            aspect="auto",
        )
        grafik_matriks.update_layout(height=370)
        st.plotly_chart(gaya_grafik(grafik_matriks), width="stretch")

    with st.expander("Lihat precision, recall, F1, dan jumlah foto"):
        st.dataframe(
            tabel_pilih.style.format(
                {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
            ),
            hide_index=True,
            width="stretch",
        )

    riwayat = pd.DataFrame(metrik.get("riwayat_latih", []))
    if not riwayat.empty:
        riwayat["tahap-epoch"] = riwayat.apply(
            lambda baris: f"T{baris['tahap']} E{baris['epoch']}", axis=1
        )
        riwayat["Akurasi latih"] = riwayat["accuracy_latih"]
        riwayat["Akurasi validasi"] = riwayat["validasi"].map(
            lambda hasil: hasil["accuracy"] if isinstance(hasil, dict) else None
        )
        grafik_latih = px.line(
            riwayat,
            x="tahap-epoch",
            y=["Akurasi latih", "Akurasi validasi"],
            markers=True,
            title="Perubahan akurasi selama pelatihan",
            labels={
                "value": "Akurasi",
                "variable": "Metrik",
                "tahap-epoch": "Tahap dan epoch",
            },
            color_discrete_map={
                "Akurasi latih": "#245b43",
                "Akurasi validasi": "#d2aa57",
            },
        )
        grafik_latih.update_yaxes(range=[0, 1], tickformat=".0%")
        grafik_latih.update_layout(height=350)
        st.plotly_chart(gaya_grafik(grafik_latih), width="stretch")

def tampilkan_klasifikasi() -> None:
    label_bagian("03 / Coba model")
    st.header("Coba model dengan satu foto")

    penyimpanan_aktif = simpan_lokal_aktif()
    kiri = st.container()
    with kiri:
        sumber_foto = st.radio(
            "Cara memasukkan foto",
            ("Scan dengan kamera", "Unggah foto"),
            horizontal=True,
            key="prediction_source",
        )
        if sumber_foto == "Scan dengan kamera":
            unggahan = st.camera_input(
                "Ambil foto sampah",
                help="Arahkan kamera ke satu objek atau tumpukan sampah.",
                key="camera_prediction",
            )
        else:
            unggahan = st.file_uploader(
                "Pilih foto sampah",
                type=("jpg", "jpeg", "png"),
                help="File diproses di memori. Penyimpanan foto memerlukan tindakan simpan lokal.",
                key="uploaded_prediction",
            )

    if unggahan is None:
        tampilkan_riwayat_lokal(penyimpanan_aktif)
        return

    try:
        gambar, tensor, gambar_asli = siapkan_gambar(unggahan)
        if ada_wajah_terdeteksi(gambar_asli):
            st.warning(
                "Wajah terdeteksi. Foto tidak diproses; ambil ulang gambar yang "
                "hanya menampilkan objek sampah."
            )
            tampilkan_riwayat_lokal(penyimpanan_aktif)
            return
        hash_foto = hashlib.sha256(unggahan.getvalue()).hexdigest()
        kunci_konfirmasi = "konfirmasi_foto_" + hash_foto
        if not st.checkbox(
            "Konfirmasi: foto hanya menampilkan sampah, tanpa wajah, plat nomor, "
            "atau informasi pribadi.",
            key=kunci_konfirmasi,
        ):
            tampilkan_riwayat_lokal(penyimpanan_aktif)
            return
        path_model = MODEL_DIR / "ecolens_mobilenetv2.onnx"
        sesi = muat_sesi_onnx(str(path_model), path_model.stat().st_mtime)
        nama_input = sesi.get_inputs()[0].name
        mulai_inferensi = time.perf_counter()
        logits = sesi.run(None, {nama_input: tensor})[0][0]
        waktu_inferensi_ms = (time.perf_counter() - mulai_inferensi) * 1000
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as kesalahan:
        st.error(f"Foto tidak dapat diperiksa atau diklasifikasikan: {kesalahan}")
        return

    if not np.isfinite(logits).all():
        st.error("Model menghasilkan nilai yang tidak valid untuk gambar ini.")
        return
    probabilitas = np.exp(logits - np.max(logits))
    probabilitas = probabilitas / probabilitas.sum()
    urutan = np.argsort(probabilitas)[::-1]
    prediksi = KELAS[int(urutan[0])]
    confidence = float(probabilitas[urutan[0]])
    waktu_per_foto = st.session_state.get("waktu_inferensi_per_foto", {})
    waktu_per_foto[hash_foto] = waktu_inferensi_ms
    st.session_state["waktu_inferensi_per_foto"] = waktu_per_foto

    st.markdown("---")
    st.subheader("Hasil prediksi")
    hasil_kiri, hasil_tengah, hasil_kanan = st.columns([0.9, 0.8, 1.3])
    hasil_kiri.image(gambar, caption="Foto yang diproses", width=280)
    hasil_tengah.markdown(
        '<div class="eco-result-label">LABEL TERATAS</div>'
        f'<div class="eco-result-value">{escape(prediksi)}</div>',
        unsafe_allow_html=True,
    )
    hasil_tengah.metric("Skor softmax", f"{confidence:.1%}")
    hasil_tengah.metric("Waktu inferensi model", f"{waktu_inferensi_ms:.1f} ms")
    if confidence < AMBANG_CONFIDENCE:
        hasil_tengah.warning(
            f"Skor di bawah ambang {AMBANG_CONFIDENCE:.0%}. "
            "Periksa kembali secara manual."
        )

    tabel_skor = pd.DataFrame(
        [
            {
                "Kelas": KELAS[int(indeks)],
                "Skor softmax": float(probabilitas[indeks]),
            }
            for indeks in urutan[:3]
        ]
    )
    grafik_skor = px.bar(
        tabel_skor.sort_values("Skor softmax"),
        x="Skor softmax",
        y="Kelas",
        orientation="h",
        range_x=[0, 1],
        color="Kelas",
        color_discrete_map=WARNA_KELAS,
        title="Tiga skor teratas",
        labels={"Skor softmax": "Skor model"},
    )
    grafik_skor.update_layout(showlegend=False, height=250)
    hasil_kanan.plotly_chart(gaya_grafik(grafik_skor), width="stretch")

    waktu_hasil = datetime.now().astimezone().isoformat(timespec="seconds")
    csv_hasil = pd.DataFrame(
        [
            {
                "waktu_lokal": waktu_hasil,
                "kelas_prediksi": prediksi,
                "skor_softmax": confidence,
                "tiga_teratas": json.dumps(
                    tabel_skor.to_dict(orient="records"), ensure_ascii=False
                ),
            }
        ]
    ).to_csv(index=False).encode("utf-8-sig")
    tombol_unduh, tombol_simpan = st.columns(2)
    tombol_unduh.download_button(
        "Unduh hasil (CSV)",
        data=csv_hasil,
        file_name="hasil_klasifikasi_ecolens.csv",
        mime="text/csv",
        key="download_current_prediction",
        use_container_width=True,
    )
    if penyimpanan_aktif:
        if tombol_simpan.button(
            "Simpan hasil + foto lokal",
            key="save_current_prediction",
            use_container_width=True,
        ):
            try:
                simpan_hasil_lokal(
                    gambar,
                    prediksi,
                    confidence,
                    tabel_skor.to_dict(orient="records"),
                )
            except (OSError, sqlite3.Error) as kesalahan:
                st.error(f"Hasil lokal gagal disimpan: {kesalahan}")
            else:
                st.success("Hasil dan foto yang diproses tersimpan di komputer tim.")
    else:
        tombol_simpan.button(
            "Simpan lokal (hanya localhost)",
            key="save_current_prediction_disabled",
            disabled=True,
            use_container_width=True,
        )
    tampilkan_riwayat_lokal(penyimpanan_aktif)


def tampilkan_riwayat_lokal(penyimpanan_aktif: bool) -> None:
    if not penyimpanan_aktif:
        return

    st.markdown("---")
    st.subheader("Riwayat lokal tim")
    try:
        riwayat = baca_riwayat_lokal()
    except sqlite3.Error as kesalahan:
        st.error(f"Riwayat lokal tidak dapat dibaca: {kesalahan}")
        return
    if st.session_state.pop("riwayat_lokal_dihapus", False):
        st.success("Riwayat dan foto lokal telah dihapus.")
    if not riwayat:
        st.info("Belum ada hasil yang disimpan.")
        return

    tabel_riwayat = pd.DataFrame(
        [
            {
                "Waktu lokal": baris["waktu_lokal"],
                "Prediksi": baris["kelas_prediksi"],
                "Skor softmax": baris["skor"],
                "Foto tersimpan": baris["nama_foto"],
            }
            for baris in riwayat[:50]
        ]
    )
    st.dataframe(
        tabel_riwayat.style.format({"Skor softmax": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )

    semua_riwayat = pd.DataFrame(
        [
            {
                "Waktu lokal": baris["waktu_lokal"],
                "Prediksi": baris["kelas_prediksi"],
                "Skor softmax": baris["skor"],
                "Foto tersimpan": baris["nama_foto"],
            }
            for baris in riwayat
        ]
    )
    bytes_csv = semua_riwayat.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Unduh riwayat lokal (CSV)",
        data=bytes_csv,
        file_name="riwayat_klasifikasi_ecolens.csv",
        mime="text/csv",
        key="download_local_history",
    )
    with st.expander("Lihat tiga foto tersimpan terbaru"):
        kolom_foto = st.columns(min(3, len(riwayat)))
        for kolom, baris in zip(kolom_foto, riwayat[:3]):
            path_foto = LOCAL_HISTORY_DIR / str(baris["nama_foto"])
            if path_foto.is_file():
                kolom.image(
                    str(path_foto),
                    caption=f"{baris['kelas_prediksi']} · {float(baris['skor']):.1%}",
                    use_container_width=True,
                )
            else:
                kolom.warning(f"Foto lokal tidak ditemukan: {baris['nama_foto']}")

    konfirmasi_hapus = st.checkbox(
        "Saya yakin ingin menghapus semua riwayat dan foto lokal.",
        key="confirm_delete_local_history",
    )
    if st.button(
        "Hapus semua riwayat lokal",
        disabled=not konfirmasi_hapus,
        key="delete_local_history",
    ):
        try:
            hapus_riwayat_lokal()
        except (OSError, sqlite3.Error) as kesalahan:
            st.error(f"Riwayat lokal gagal dihapus: {kesalahan}")
        else:
            st.session_state["riwayat_lokal_dihapus"] = True
            st.rerun()

def tampilkan_kandidat(analisis: dict) -> None:
    label_bagian("04 / Tindak lanjut")
    st.header("Saran pengembangan model")
    for nomor, kandidat in enumerate(analisis["kandidat_rekomendasi"], start=1):
        with st.container(border=True):
            st.subheader(f"{nomor}. {kandidat['judul']}")
            st.markdown(f"**Bukti uji:** {kandidat['dasar']}")
            st.write(kandidat["usulan_kandidat"])


def tampilkan_info_model(analisis: dict) -> None:
    label_bagian("05 / Informasi")
    st.header("Sumber data dan model")
    sumber = analisis["sumber"]
    kolom_data, kolom_model = st.columns(2)
    with kolom_data:
        st.subheader("Dataset")
        st.markdown(f"**{sumber['nama']}**")
        st.markdown(f"[Buka sumber dataset]({sumber['url']})")
        st.write(f"Lisensi yang dicantumkan sumber: {sumber['lisensi']}.")
        st.write(sumber["peran"])
    with kolom_model:
        st.subheader("Model")
        st.write("MobileNetV2 dengan bobot awal ImageNet.")
        st.write("Enam label keluaran: " + ", ".join(KELAS) + ".")
        st.write("Foto yang dipilih diproses dengan ONNX Runtime.")
        st.write(
            "Pemeriksaan wajah menolak wajah yang terdeteksi, tetapi tidak "
            "memastikan setiap foto lain berisi sampah."
        )
        path_model = MODEL_DIR / "ecolens_mobilenetv2.onnx"
        try:
            ukuran_model = path_model.stat().st_size
        except OSError as kesalahan:
            st.error(f"Ukuran model tidak dapat dibaca: {kesalahan}")
        else:
            st.metric("Ukuran model ONNX", f"{ukuran_model / 1_000_000:.2f} MB")
        waktu_sesi = st.session_state.get("waktu_inferensi_per_foto", {})
        if waktu_sesi:
            rata_rata_ms = sum(waktu_sesi.values()) / len(waktu_sesi)
            st.metric(
                f"Rata-rata inferensi sesi ({len(waktu_sesi)} foto)",
                f"{rata_rata_ms:.1f} ms",
            )

    st.subheader("Pemetaan folder ke label")
    pemetaan = analisis["dataset_asal"].get("pemetaan_per_folder", {})
    tabel_pemetaan = pd.DataFrame(
        [
            {"Folder sumber": folder, "Label model": label}
            for folder, label in pemetaan.items()
        ]
    )
    st.dataframe(tabel_pemetaan, hide_index=True, width="stretch")

def main() -> None:
    st.set_page_config(page_title="EcoLens AI | Klasifikasi sampah", layout="wide")
    st.markdown(GAYA, unsafe_allow_html=True)
    try:
        logo_svg = (PROJECT_ROOT / "assets" / "ecolens-logo.svg").read_text(
            encoding="utf-8"
        )
        mark_svg = (PROJECT_ROOT / "assets" / "ecolens-mark.svg").read_text(
            encoding="utf-8"
        )
    except OSError as kesalahan:
        st.error(f"Logo dashboard tidak dapat dimuat: {kesalahan}")
        st.stop()

    st.sidebar.markdown(
        f'<div class="eco-sidebar-brand">{logo_svg}</div>',
        unsafe_allow_html=True,
    )
    halaman = st.sidebar.radio(
        "Navigasi",
        (
            "Ikhtisar data",
            "Evaluasi model",
            "Coba klasifikasi",
            "Tindak lanjut",
            "Tentang model",
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Klasifikasi visual · enam kelas")

    st.markdown(
        '<div class="eco-brandline">'
        f'<span class="eco-brand-mark">{mark_svg}</span>'
        "<span>Computer vision / klasifikasi sampah</span>"
        '<span class="eco-pill">6 kelas</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("EcoLens AI")
    st.caption("Klasifikasi visual sampah untuk mendukung edukasi pemilahan.")

    try:
        analisis, metrik, manifest = muat_data()
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError) as kesalahan:
        st.error(f"Dashboard belum dapat memuat laporan proyek: {kesalahan}")
        st.stop()

    if halaman == "Ikhtisar data":
        tampilkan_ikhtisar(analisis, metrik, manifest)
    elif halaman == "Evaluasi model":
        tampilkan_evaluasi(metrik, manifest)
    elif halaman == "Coba klasifikasi":
        tampilkan_klasifikasi()
    elif halaman == "Tindak lanjut":
        tampilkan_kandidat(analisis)
    else:
        tampilkan_info_model(analisis)

    st.markdown(
        '<div class="eco-footer">EcoLens AI · Klasifikasi visual sampah</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
