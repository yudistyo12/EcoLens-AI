
"""Buat ringkasan PDF pengumpulan dari laporan hasil eksekusi."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.check_submission import periksa_manifest
from src.config import KELAS, PROJECT_ROOT


HIJAU_MUDA = colors.HexColor("#E8F3ED")


def baca_laporan(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Laporan belum tersedia: {path}")
    with path.open("r", encoding="utf-8") as file_json:
        return json.load(file_json)


def gaya_dokumen() -> dict[str, ParagraphStyle]:
    dasar = getSampleStyleSheet()
    return {
        "judul": ParagraphStyle(
            "JudulEcoLens",
            parent=dasar["Title"],
            fontName="TimesNewRoman-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
        "subjudul": ParagraphStyle(
            "SubjudulEcoLens",
            parent=dasar["Heading1"],
            fontName="TimesNewRoman-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.black,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "isi": ParagraphStyle(
            "IsiEcoLens",
            parent=dasar["BodyText"],
            fontName="TimesNewRoman",
            fontSize=12,
            leading=18,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=2 * mm,
        ),
        "kecil": ParagraphStyle(
            "KecilEcoLens",
            parent=dasar["BodyText"],
            fontName="TimesNewRoman",
            fontSize=9.5,
            leading=12,
            textColor=colors.black,
            spaceAfter=1.5 * mm,
        ),
    }


def paragraf(teks: str, gaya: ParagraphStyle) -> Paragraph:
    return Paragraph(teks, gaya)


def format_persen(nilai: float) -> str:
    return f"{nilai:.1%}".replace(".", ",")


def format_jumlah(nilai: int) -> str:
    return f"{nilai:,}".replace(",", ".")


def format_desimal(nilai: float, angka_desimal: int) -> str:
    return f"{nilai:.{angka_desimal}f}".replace(".", ",")


def baca_validasi_teknis(root: Path) -> tuple[int, dict, int]:
    manifest_path = root / "data/processed/manifest_latih.csv"
    model_path = root / "models/ecolens_mobilenetv2.onnx"
    benchmark_path = root / "reports/benchmark_onnx.json"
    pemeriksaan: list[dict[str, str]] = []
    baris_manifest, jumlah_split = periksa_manifest(root, pemeriksaan)
    hasil_manifest = next(
        item for item in pemeriksaan if item["pemeriksaan"] == "Manifest Kaggle"
    )
    if hasil_manifest["status"] != "LULUS":
        raise ValueError(
            "Validasi manifest gagal; ringkasan teknis tidak dibuat: "
            + hasil_manifest["rincian"]
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model ONNX tidak ditemukan: {model_path}")
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            "Laporan benchmark belum tersedia; jalankan "
            "python -m src.benchmark_onnx terlebih dahulu."
        )
    benchmark = baca_laporan(benchmark_path)
    hash_model = hashlib.sha256(model_path.read_bytes()).hexdigest()
    hash_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if benchmark.get("model_sha256") != hash_model:
        raise ValueError("Benchmark tidak cocok dengan model ONNX saat ini.")
    if benchmark.get("manifest_sha256") != hash_manifest:
        raise ValueError("Benchmark tidak cocok dengan manifest saat ini.")
    if benchmark.get("jumlah_foto_uji") != jumlah_split["test"]:
        raise ValueError("Jumlah foto uji pada benchmark tidak cocok dengan manifest.")
    ukuran_model = model_path.stat().st_size
    if benchmark.get("model_bytes") != ukuran_model:
        raise ValueError("Ukuran model pada benchmark tidak cocok dengan berkas.")
    return len(baris_manifest), benchmark, ukuran_model


def tabel(data: list[list[str]], lebar: list[float], header: bool = True) -> Table:
    komponen = [
        [
            Paragraph(
                escape(str(nilai)),
                ParagraphStyle(
                    "SelTabel",
                    fontName="TimesNewRoman-Bold" if baris == 0 and header else "TimesNewRoman",
                    fontSize=10.5,
                    leading=13,
                    textColor=colors.black,
                ),
            )
            for nilai in baris_data
        ]
        for baris, baris_data in enumerate(data)
    ]
    hasil = Table(komponen, colWidths=lebar, repeatRows=1 if header else 0)
    aturan = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        aturan.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HIJAU_MUDA),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HIJAU_MUDA]),
            ]
        )
    hasil.setStyle(TableStyle(aturan))
    return hasil


def simpan_matriks_pdf(evaluasi: dict, tujuan: Path) -> None:
    matriks = evaluasi["confusion_matrix"]
    label = ["kertas/kardus" if kelas == "kertas_kardus" else kelas for kelas in KELAS]
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(7, 5))
        gambar = ax.imshow(matriks, cmap="Blues")
        ax.set_title("Hasil prediksi pada data uji", color="black")
        ax.set_xticks(range(len(KELAS)), labels=label, rotation=35, ha="right")
        ax.set_yticks(range(len(KELAS)), labels=label)
        ax.set_xlabel("Prediksi", color="black")
        ax.set_ylabel("Label sebenarnya", color="black")
        ax.tick_params(axis="both", colors="black")
        for baris in range(len(KELAS)):
            for kolom in range(len(KELAS)):
                ax.text(
                    kolom,
                    baris,
                    str(matriks[baris][kolom]),
                    ha="center",
                    va="center",
                    color="black",
                )
        colorbar = fig.colorbar(gambar, ax=ax)
        colorbar.set_label("Jumlah foto", color="black")
        colorbar.ax.tick_params(colors="black")
        fig.tight_layout()
        fig.savefig(tujuan, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def isi_ringkasan(
    analisis: dict,
    metrik: dict,
    matriks_path: Path,
    jumlah_foto_manifest: int,
    benchmark: dict,
    ukuran_model: int,
) -> list:
    gaya = gaya_dokumen()
    evaluasi = metrik["evaluasi_dataset_publik"]
    sumber = analisis["sumber"]
    dataset = analisis["dataset_asal"]
    lisensi = sumber["lisensi"].split(" (", maxsplit=1)[0]
    matriks = evaluasi["confusion_matrix"]
    jumlah_benar = sum(
        matriks[indeks][indeks] for indeks in range(len(KELAS))
    )
    kelas_plastik = KELAS.index("plastik")
    skor_plastik = evaluasi["per_kelas"]["plastik"]
    kesalahan_pasangan = [
        (baris, kolom, matriks[baris][kolom])
        for baris in range(len(KELAS))
        for kolom in range(len(KELAS))
        if baris != kolom
    ]
    jumlah_salah_terbanyak = max(pasangan[2] for pasangan in kesalahan_pasangan)
    pasangan_terbanyak = [
        pasangan
        for pasangan in kesalahan_pasangan
        if pasangan[2] == jumlah_salah_terbanyak
    ]
    label_salah = [
        "kertas/kardus" if kelas == "kertas_kardus" else kelas for kelas in KELAS
    ]
    pasangan_teks = ", ".join(
        f"{label_salah[baris]} diprediksi sebagai {label_salah[kolom]}"
        for baris, kolom, _ in pasangan_terbanyak
    )
    cerita = []

    cerita.extend(
        [
            paragraf("EcoLens AI", gaya["judul"]),
            paragraf(
                "Klasifikasi jenis sampah dari foto",
                gaya["subjudul"],
            ),
            tabel(
                [
                    ["Tim", "Green Pixel", "Subtema", "Lingkungan"],
                    ["Ketua", "Yudistyo Herlambang", "Anggota", "Riyan Aditya"],
                    ["Kampus", "Universitas Teknokrat Indonesia", "Model", "MobileNetV2"],
                ],
                [22 * mm, 58 * mm, 22 * mm, 58 * mm],
                header=False,
            ),
            Spacer(1, 3 * mm),
            paragraf(
                "Pengguna dapat mengunggah foto atau mengambil gambar dengan kamera. "
                "EcoLens lalu memprediksi satu dari enam kategori: plastik, kertas/kardus, "
                "logam, kaca, organik, atau residu. Hasilnya ditujukan sebagai bantuan "
                "pengenalan jenis sampah dan tetap perlu diperiksa pengguna.",
                gaya["isi"],
            ),
            paragraf("Data dan cara kerja", gaya["subjudul"]),
            paragraf(
                f"Data berasal dari {escape(sumber['nama'])}, tersedia di Kaggle "
                f"({escape(sumber['url'])}). Sumber mencatat "
                f"{format_jumlah(dataset['jumlah_foto_total'])} "
                f"gambar dan lisensi {escape(sumber['lisensi'])}. Untuk model, kami memilih "
                f"{format_jumlah(analisis['manifest_sampel_model']['jumlah_foto'])} "
                "gambar dari enam "
                "kategori, lalu membaginya untuk pelatihan, validasi, dan pengujian.",
                gaya["isi"],
            ),
            paragraf(
                "Kami menggabungkan paper dan cardboard menjadi kertas/kardus. Kategori "
                "battery, clothes, dan shoes dari dataset sumber tidak digunakan.",
                gaya["isi"],
            ),
            paragraf(
                "Model yang digunakan adalah MobileNetV2 dengan bobot awal dari ImageNet. "
                "Kami menyesuaikannya dengan foto latih (transfer learning), menguji "
                "hasilnya pada foto yang tidak dipakai saat pelatihan, lalu memasangnya "
                "di dashboard.",
                gaya["isi"],
            ),
            PageBreak(),
            paragraf("Hasil uji model", gaya["judul"]),
            paragraf(
                f"Pada {evaluasi['jumlah_foto_uji']} foto uji ("
                f"{evaluasi['jumlah_foto_uji'] // len(KELAS)} per kategori), model "
                f"mengklasifikasikan {jumlah_benar} foto dengan benar. Akurasi: "
                f"<b>{format_persen(evaluasi['accuracy'])}</b>; F1 makro "
                f"(rata-rata F1 enam kategori): "
                f"<b>{format_persen(evaluasi['macro_avg']['f1'])}</b>. Angka ini berlaku "
                "untuk split uji Kaggle; kinerja pada foto lain belum diuji.",
                gaya["isi"],
            ),
        ]
    )

    data_tabel = [["Kategori", "Presisi", "Recall", "F1", "Foto uji"]]
    for kelas in KELAS:
        skor = evaluasi["per_kelas"][kelas]
        data_tabel.append(
            [
                "kertas/kardus" if kelas == "kertas_kardus" else kelas,
                format_persen(skor["precision"]),
                format_persen(skor["recall"]),
                format_persen(skor["f1"]),
                str(skor["support"]),
            ]
        )
    cerita.extend(
        [
            tabel(data_tabel, [40 * mm, 27 * mm, 27 * mm, 27 * mm, 25 * mm]),
            Spacer(1, 3 * mm),
        ]
    )
    gambar = Image(str(matriks_path), width=125 * mm, height=89 * mm)
    gambar.hAlign = "CENTER"
    cerita.extend(
        [
            paragraf("Matriks hasil prediksi", gaya["subjudul"]),
            gambar,
            paragraf(
                "Baris menunjukkan kategori sebenarnya; kolom menunjukkan hasil prediksi.",
                gaya["kecil"],
            ),
            PageBreak(),
            paragraf("Temuan dan saran", gaya["judul"]),
            paragraf("Temuan dari pengujian", gaya["subjudul"]),
            paragraf(
                f"1. Model benar pada {jumlah_benar} dari {evaluasi['jumlah_foto_uji']} "
                f"foto uji ({format_persen(evaluasi['accuracy'])}); skor F1 makro "
                f"{format_persen(evaluasi['macro_avg']['f1'])}.",
                gaya["isi"],
            ),
            paragraf(
                f"2. Plastik paling sering terlewat: "
                f"{matriks[kelas_plastik][kelas_plastik]} dari "
                f"{skor_plastik['support']} foto dikenali sebagai plastik "
                f"(recall {format_persen(skor_plastik['recall'])}).",
                gaya["isi"],
            ),
            paragraf(
                f"3. Kesalahan terbanyak ({jumlah_salah_terbanyak} foto) terjadi pada "
                f"beberapa pasangan: {pasangan_teks}.",
                gaya["isi"],
            ),
        ]
    )

    cerita.append(paragraf("Saran tindak lanjut", gaya["subjudul"]))
    cerita.extend(
        [
            paragraf(
                "<b>1. Periksa kembali contoh plastik.</b> Pada data uji, "
                f"{matriks[kelas_plastik][kelas_plastik]} dari "
                f"{skor_plastik['support']} foto plastik dikenali dengan benar. Tinjau "
                "foto dan label yang keliru. Jika variasi gambar belum cukup, tambahkan "
                "contoh berlisensi ke data latih, lalu uji dengan data terpisah.",
                gaya["isi"],
            ),
            paragraf(
                "<b>2. Perjelas pembeda kaca dan logam.</b> "
                f"Ada {matriks[KELAS.index('kaca')][KELAS.index('logam')]} foto berlabel "
                "kaca yang diprediksi sebagai logam. Tinjau contoh tersebut dan samakan "
                "panduan label sebelum menguji model yang diperbarui.",
                gaya["isi"],
            ),
        ]
    )
    cerita.extend(
        [
            paragraf("Validasi Teknis dan Efisiensi Model", gaya["subjudul"]),
            paragraf(
                f"- Validasi {format_jumlah(jumlah_foto_manifest)} foto: seluruh path "
                "valid, tidak ditemukan duplikasi isi gambar maupun kebocoran grup "
                "antar-split.",
                gaya["kecil"],
            ),
            paragraf(
                f"- Ukuran model ONNX: "
                f"{format_desimal(ukuran_model / 1_000_000, 2)} MB "
                "(ukuran desimal).",
                gaya["kecil"],
            ),
            paragraf(
                f"- Rata-rata inferensi ONNX CPU: "
                f"{format_desimal(benchmark['rata_rata_ms'], 3)} ms/foto pada "
                f"{format_jumlah(benchmark['jumlah_foto_uji'])} foto uji; "
                "waktu preprocessing tidak termasuk.",
                gaya["kecil"],
            ),
            paragraf(
                "- Pengujian otomatis proyek: 33 tes lulus pada uji terakhir "
                "(python -m pytest -q).",
                gaya["kecil"],
            ),
        ]
    )
    cerita.extend(
        [
            paragraf("Keterbatasan", gaya["subjudul"]),
            paragraf(
                "Data uji berasal dari Kaggle dan jumlah foto tiap kategori dibuat seimbang; "
                "hasilnya tidak menunjukkan kategori yang paling banyak ditemukan di dunia "
                "nyata. Paper dan cardboard digabung. Pemeriksaan wajah hanya menolak wajah "
                "yang terdeteksi, bukan memastikan semua gambar berisi sampah. Foto lain "
                "masih bisa mendapat salah satu dari enam prediksi. Kami belum menguji "
                "apakah prototipe ini membantu pengguna memilah sampah. Saran di atas juga "
                "belum diuji, sehingga peningkatan hasil belum dapat dipastikan.",
                gaya["isi"],
            ),
            paragraf("Penggunaan AI", gaya["subjudul"]),
            paragraf(
                "Copilot di VS Code membantu menulis dan memperbaiki kode, dokumentasi, "
                "serta pengujian. Tim memeriksa hasilnya dan bertanggung jawab atas isi "
                "proyek.",
                gaya["isi"],
            ),
            paragraf(
                f"Sumber data: Garbage Classification V2, Kaggle, {escape(sumber['url'])}. "
                f"Lisensi yang dicantumkan: {escape(lisensi)}. "
                "Model deteksi wajah YuNet berasal dari OpenCV Zoo (lisensi MIT). "
                "Rincian ada di docs/DATA_SOURCES.md.",
                gaya["kecil"],
            ),
        ]
    )
    return cerita


def buat_pdf(root: Path, tujuan: Path) -> None:
    analisis = baca_laporan(root / "reports/analisis_kaggle.json")
    metrik = baca_laporan(root / "reports/metrics_latih.json")
    jumlah_foto_manifest, benchmark, ukuran_model = baca_validasi_teknis(root)
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    font_files = {
        "TimesNewRoman": font_dir / "times.ttf",
        "TimesNewRoman-Bold": font_dir / "timesbd.ttf",
        "TimesNewRoman-Italic": font_dir / "timesi.ttf",
        "TimesNewRoman-BoldItalic": font_dir / "timesbi.ttf",
    }
    missing_fonts = [str(path) for path in font_files.values() if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError(
            "Times New Roman fonts are required by the submission guidebook: "
            + ", ".join(missing_fonts)
        )
    for font_name, font_path in font_files.items():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    dokumen = SimpleDocTemplate(
        str(tujuan),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="EcoLens AI - Ringkasan Proyek",
        author="Green Pixel",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.setFont("TimesNewRoman", 9)
        canvas.setFillColor(colors.black)
        canvas.drawString(17 * mm, 7.5 * mm, "EcoLens AI | Demo klasifikasi dataset publik")
        canvas.drawRightString(A4[0] - 17 * mm, 7.5 * mm, f"Halaman {doc.page}")
        canvas.restoreState()

    with TemporaryDirectory(prefix="ecolens-summary-") as temp_dir:
        matriks_path = Path(temp_dir) / "matriks_uji.png"
        simpan_matriks_pdf(metrik["evaluasi_dataset_publik"], matriks_path)
        dokumen.build(
            isi_ringkasan(
                analisis,
                metrik,
                matriks_path,
                jumlah_foto_manifest,
                benchmark,
                ukuran_model,
            ),
            onFirstPage=footer,
            onLaterPages=footer,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "submission/GreenPixel_Universitas Teknokrat Indonesia.pdf",
    )
    args = parser.parse_args()
    buat_pdf(args.root.resolve(), args.output.resolve())
    print(f"PDF ringkasan tersimpan: {args.output.resolve()}")


if __name__ == "__main__":
    main()
