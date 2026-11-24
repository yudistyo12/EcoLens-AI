"""Fungsi kecil untuk membaca nama foto dan mengelompokkan waktu."""

import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

from src.config import KELAS, RENTANG_WAKTU


@dataclass(frozen=True)
class NamaFotoLatih:
    kelas: str
    id_objek: str
    nomor_foto: int

    @property
    def group_id(self) -> str:
        return f"{self.kelas}_{self.id_objek}"


@dataclass(frozen=True)
class NamaFotoAudit:
    kode_lokasi: str
    tanggal: str
    jam: str
    nomor: int


_KELAS_LATIH = "|".join(re.escape(kelas) for kelas in sorted(KELAS, key=len, reverse=True))
_POLA_LATIH = re.compile(
    rf"^(?P<kelas>{_KELAS_LATIH})_(?P<id_objek>[^_]+)_(?P<nomor>\d+)\.jpe?g$",
    re.IGNORECASE,
)
_POLA_AUDIT = re.compile(
    r"^(?P<lokasi>[A-Za-z0-9]+)_(?P<tanggal>\d{8})_"
    r"(?P<jam>\d{4})_(?P<nomor>\d+)\.jpe?g$",
    re.IGNORECASE,
)


def parse_nama_foto_latih(nama_file: str | Path) -> NamaFotoLatih | None:
    """Parse pola <kelas>_<idobjek>_<nomor>.jpg; None bila tidak cocok."""
    hasil = _POLA_LATIH.fullmatch(Path(nama_file).name)
    if hasil is None:
        return None

    kelas = hasil.group("kelas").lower()
    return NamaFotoLatih(
        kelas=kelas,
        id_objek=hasil.group("id_objek"),
        nomor_foto=int(hasil.group("nomor")),
    )


def parse_nama_foto_audit(nama_file: str | Path) -> NamaFotoAudit | None:
    """Parse pola <lokasi>_<YYYYMMDD>_<HHMM>_<nomor>.jpg."""
    hasil = _POLA_AUDIT.fullmatch(Path(nama_file).name)
    if hasil is None:
        return None

    tanggal = hasil.group("tanggal")
    jam = hasil.group("jam")
    try:
        datetime.strptime(f"{tanggal}{jam}", "%Y%m%d%H%M")
    except ValueError:
        return None

    return NamaFotoAudit(
        kode_lokasi=hasil.group("lokasi").upper(),
        tanggal=tanggal,
        jam=jam,
        nomor=int(hasil.group("nomor")),
    )


def tentukan_waktu(jam: str | time) -> str:
    """Kembalikan pagi/siang/sore, atau di luar jadwal yang ditentukan."""
    if isinstance(jam, str):
        teks_jam = jam.strip()
        pola = "%H:%M" if ":" in teks_jam else "%H%M"
        try:
            jam_objek = datetime.strptime(teks_jam, pola).time()
        except ValueError as kesalahan:
            raise ValueError("Jam harus berformat HHMM atau HH:MM.") from kesalahan
    elif isinstance(jam, time):
        jam_objek = jam
    else:
        raise TypeError("jam harus berupa teks atau datetime.time.")

    for nama_waktu, (mulai, selesai) in RENTANG_WAKTU.items():
        if mulai <= jam_objek <= selesai:
            return nama_waktu
    return "di luar jadwal"
