from src.prepare_data import bagi_split


def test_bagi_split_menjaga_foto_satu_grup_bersama():
    baris = [
        {
            "path": f"foto_{nomor}.jpg",
            "kelas": "plastik",
            "sumber": "kampus",
            "group_id": grup,
        }
        for grup, nomor in [("objek_a", 1), ("objek_a", 2), ("objek_b", 3), ("objek_c", 4)]
    ]

    hasil = bagi_split(baris)
    split_per_grup = {}
    for item in hasil:
        split_per_grup.setdefault(item["group_id"], set()).add(item["split"])

    assert len(split_per_grup["objek_a"]) == 1
    assert len({item["split"] for item in hasil}) > 1


def test_bagi_split_dapat_diulang_dengan_seed_sama():
    baris = [
        {
            "path": f"foto_{nomor}.jpg",
            "kelas": "logam",
            "sumber": "gd_v2",
            "group_id": f"grup_{nomor}",
        }
        for nomor in range(10)
    ]

    assert bagi_split(baris, seed=42) == bagi_split(baris, seed=42)
