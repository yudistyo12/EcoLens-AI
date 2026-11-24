from datetime import time

import pytest

from src.utils import (
    parse_nama_foto_audit,
    parse_nama_foto_latih,
    tentukan_waktu,
)


def test_parse_nama_foto_latih_membaca_kelas_dan_grup_objek():
    hasil = parse_nama_foto_latih(r"data\plastik_O012_1.jpg")

    assert hasil is not None
    assert hasil.kelas == "plastik"
    assert hasil.id_objek == "O012"
    assert hasil.nomor_foto == 1
    assert hasil.group_id == "plastik_O012"


def test_parse_nama_foto_latih_mendukung_kelas_dengan_garis_bawah():
    hasil = parse_nama_foto_latih("kertas_kardus_K004_3.JPG")

    assert hasil is not None
    assert hasil.kelas == "kertas_kardus"
    assert hasil.group_id == "kertas_kardus_K004"


def test_parse_nama_foto_latih_menolak_pola_yang_tidak_valid():
    assert parse_nama_foto_latih("plastik_O012.jpg") is None
    assert parse_nama_foto_latih("kelas_tidak_dikenal_O012_1.jpg") is None


def test_parse_nama_foto_audit_membaca_nama_dan_menolak_tanggal_invalid():
    hasil = parse_nama_foto_audit("L03_20261128_0935_004.jpg")

    assert hasil is not None
    assert hasil.kode_lokasi == "L03"
    assert hasil.tanggal == "20261128"
    assert hasil.jam == "0935"
    assert hasil.nomor == 4
    assert parse_nama_foto_audit("L03_20260230_0935_004.jpg") is None


@pytest.mark.parametrize(
    ("jam", "waktu"),
    [
        ("0500", "pagi"),
        ("10:59", "pagi"),
        ("1100", "siang"),
        ("14:59", "siang"),
        ("1500", "sore"),
        ("18:59", "sore"),
        ("0459", "di luar jadwal"),
        (time(19, 0), "di luar jadwal"),
    ],
)
def test_tentukan_waktu_batas_jadwal(jam, waktu):
    assert tentukan_waktu(jam) == waktu


def test_tentukan_waktu_memberi_pesan_jika_format_tidak_valid():
    with pytest.raises(ValueError, match="HHMM atau HH:MM"):
        tentukan_waktu("pagi")
