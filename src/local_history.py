"""Penyimpanan lokal opsional untuk hasil inferensi dan gambar yang diproses."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from PIL import Image

from src.config import PROJECT_ROOT


LOCAL_HISTORY_DIR = PROJECT_ROOT / "data" / "local_history"
DATABASE_NAME = "riwayat.sqlite3"


def penyimpanan_lokal_diizinkan(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"localhost", "127.0.0.1", "::1"}


def simpan_hasil_lokal(
    gambar: Image.Image,
    kelas_prediksi: str,
    skor: float,
    tiga_teratas: list[dict[str, str | float]],
    direktori: Path = LOCAL_HISTORY_DIR,
) -> dict[str, str | float]:
    direktori.mkdir(parents=True, exist_ok=True)
    id_hasil = uuid4().hex
    nama_foto = f"{id_hasil}.jpg"
    path_foto = direktori / nama_foto
    path_sementara = direktori / f".{id_hasil}.tmp"
    path_database = direktori / DATABASE_NAME

    try:
        gambar.save(path_sementara, format="JPEG", quality=90, optimize=True)
        path_sementara.replace(path_foto)
    except OSError:
        path_sementara.unlink(missing_ok=True)
        raise

    waktu_lokal = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        with closing(sqlite3.connect(path_database)) as koneksi:
            with koneksi:
                koneksi.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediksi (
                        id TEXT PRIMARY KEY,
                        waktu_lokal TEXT NOT NULL,
                        kelas_prediksi TEXT NOT NULL,
                        skor REAL NOT NULL,
                        tiga_teratas_json TEXT NOT NULL,
                        nama_foto TEXT NOT NULL
                    )
                    """
                )
                koneksi.execute(
                    """
                    INSERT INTO prediksi (
                        id, waktu_lokal, kelas_prediksi, skor,
                        tiga_teratas_json, nama_foto
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        id_hasil,
                        waktu_lokal,
                        kelas_prediksi,
                        float(skor),
                        json.dumps(tiga_teratas, ensure_ascii=False),
                        nama_foto,
                    ),
                )
    except sqlite3.Error:
        path_foto.unlink(missing_ok=True)
        raise

    return {
        "id": id_hasil,
        "waktu_lokal": waktu_lokal,
        "kelas_prediksi": kelas_prediksi,
        "skor": float(skor),
        "tiga_teratas_json": json.dumps(tiga_teratas, ensure_ascii=False),
        "nama_foto": nama_foto,
    }


def baca_riwayat_lokal(
    direktori: Path = LOCAL_HISTORY_DIR, batas: int | None = None
) -> list[dict[str, str | float]]:
    path_database = direktori / DATABASE_NAME
    if not path_database.is_file():
        return []

    with closing(sqlite3.connect(path_database)) as koneksi:
        koneksi.row_factory = sqlite3.Row
        query = """
            SELECT id, waktu_lokal, kelas_prediksi, skor,
                   tiga_teratas_json, nama_foto
            FROM prediksi
            ORDER BY waktu_lokal DESC
        """
        if batas is None:
            baris = koneksi.execute(query).fetchall()
        else:
            baris = koneksi.execute(query + " LIMIT ?", (batas,)).fetchall()
    return [dict(item) for item in baris]


def hapus_riwayat_lokal(direktori: Path = LOCAL_HISTORY_DIR) -> None:
    path_database = direktori / DATABASE_NAME
    if not path_database.is_file():
        return

    with closing(sqlite3.connect(path_database)) as koneksi:
        nama_foto = [
            baris[0]
            for baris in koneksi.execute("SELECT nama_foto FROM prediksi").fetchall()
        ]
    for nama in nama_foto:
        if Path(nama).name == nama and nama.endswith(".jpg"):
            (direktori / nama).unlink(missing_ok=True)
    path_database.unlink(missing_ok=True)
