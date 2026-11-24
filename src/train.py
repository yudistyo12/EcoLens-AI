"""Latih MobileNetV2 dan simpan metrik evaluasi yang terpisah."""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.config import KELAS, MODEL_DIR, PROCESSED_DIR, REPORT_DIR, SEED


UKURAN_GAMBAR = 224
RATA_RATA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


class DatasetManifest(Dataset):
    """Membaca foto sesuai path dan label yang tercatat pada manifest."""

    def __init__(self, baris: list[dict[str, str]], transformasi):
        self.baris = baris
        self.transformasi = transformasi

    def __len__(self) -> int:
        return len(self.baris)

    def __getitem__(self, indeks: int):
        baris = self.baris[indeks]
        with Image.open(baris["_path_absolut"]) as gambar:
            gambar.getexif().clear()
            gambar_rgb = gambar.convert("RGB")
        return self.transformasi(gambar_rgb), KELAS.index(baris["kelas"])


def atur_seed(seed: int) -> None:
    """Atur seed Python, NumPy, dan PyTorch agar proses dapat diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def baca_manifest(path_manifest: Path, max_per_kelas: int, seed: int) -> list[dict[str, str]]:
    """Baca manifest, validasi kelas/path, dan batasi data secara reproducible."""
    if not path_manifest.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {path_manifest}")

    baris_terpilih: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with path_manifest.open("r", newline="", encoding="utf-8") as file_csv:
        pembaca = csv.DictReader(file_csv)
        kolom_wajib = {"path", "kelas", "sumber", "group_id", "split"}
        if not kolom_wajib.issubset(pembaca.fieldnames or []):
            raise ValueError(f"Manifest harus memiliki kolom: {sorted(kolom_wajib)}")

        for baris in pembaca:
            if baris["kelas"] not in KELAS:
                raise ValueError(f"Kelas tidak dikenal pada manifest: {baris['kelas']}")
            if baris["split"] not in {"train", "val", "test"}:
                raise ValueError(f"Split tidak dikenal: {baris['split']}")
            path_foto = (path_manifest.parent.parent.parent / baris["path"]).resolve()
            if not path_foto.is_file():
                raise FileNotFoundError(f"Foto pada manifest tidak ditemukan: {path_foto}")
            baris["_path_absolut"] = str(path_foto)
            kunci = (baris["sumber"], baris["kelas"], baris["split"])
            baris_terpilih[kunci].append(baris)

    hasil = []
    rng = random.Random(seed)
    for kunci in sorted(baris_terpilih):
        contoh = baris_terpilih[kunci]
        rng.shuffle(contoh)
        if max_per_kelas > 0:
            contoh = contoh[:max_per_kelas]
        hasil.extend(contoh)
    return hasil


def buat_transformasi():
    """Siapkan augmentasi ringan untuk latih dan ukuran tetap untuk evaluasi."""
    transformasi_latih = transforms.Compose(
        [
            transforms.Resize((UKURAN_GAMBAR, UKURAN_GAMBAR)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(RATA_RATA_IMAGENET, STD_IMAGENET),
        ]
    )
    transformasi_evaluasi = transforms.Compose(
        [
            transforms.Resize((UKURAN_GAMBAR, UKURAN_GAMBAR)),
            transforms.ToTensor(),
            transforms.Normalize(RATA_RATA_IMAGENET, STD_IMAGENET),
        ]
    )
    return transformasi_latih, transformasi_evaluasi


def buat_loader(
    baris: list[dict[str, str]],
    transformasi,
    batch_size: int,
    acak: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        DatasetManifest(baris, transformasi),
        batch_size=batch_size,
        shuffle=acak,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def buat_bobot_kelas(baris_latih: list[dict[str, str]]) -> torch.Tensor:
    """Hitung bobot invers frekuensi agar kelas minoritas lebih diperhatikan."""
    jumlah = Counter(baris["kelas"] for baris in baris_latih)
    banyak_kelas = len(KELAS)
    total = len(baris_latih)
    bobot = [
        total / (banyak_kelas * jumlah[kelas]) if jumlah[kelas] else 0.0
        for kelas in KELAS
    ]
    return torch.tensor(bobot, dtype=torch.float32)


def atur_parameter_model(model: nn.Module, tahap: int) -> None:
    """Bekukan backbone pada tahap awal, lalu buka beberapa blok terakhir."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if tahap == 2:
        for blok in list(model.features.children())[-3:]:
            for parameter in blok.parameters():
                parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def jalankan_satu_epoch(
    model: nn.Module,
    loader: DataLoader,
    fungsi_loss,
    device: torch.device,
    latih: bool,
    optimizer=None,
) -> tuple[float, float]:
    model.train(latih)
    if latih:
        for modul in model.modules():
            if isinstance(modul, nn.BatchNorm2d):
                if not any(parameter.requires_grad for parameter in modul.parameters()):
                    modul.eval()

    jumlah_loss = 0.0
    jumlah_benar = 0
    jumlah_data = 0
    for gambar, label in loader:
        gambar = gambar.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        if latih:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(latih):
            keluaran = model(gambar)
            loss = fungsi_loss(keluaran, label)
            if latih:
                loss.backward()
                optimizer.step()

        ukuran_batch = label.size(0)
        jumlah_loss += loss.item() * ukuran_batch
        jumlah_benar += (keluaran.argmax(dim=1) == label).sum().item()
        jumlah_data += ukuran_batch

    return jumlah_loss / jumlah_data, jumlah_benar / jumlah_data


def evaluasi(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    label_asli = []
    label_prediksi = []
    with torch.inference_mode():
        for gambar, label in loader:
            keluaran = model(gambar.to(device, non_blocking=True))
            label_asli.extend(label.tolist())
            label_prediksi.extend(keluaran.argmax(dim=1).cpu().tolist())
    return label_asli, label_prediksi


def buat_laporan_metrik(
    label_asli: list[int], label_prediksi: list[int]
) -> dict[str, object] | None:
    """Hitung metrik hanya jika ada foto uji aktual."""
    if not label_asli:
        return None

    indeks_kelas = list(range(len(KELAS)))
    rincian = classification_report(
        label_asli,
        label_prediksi,
        labels=indeks_kelas,
        target_names=list(KELAS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "jumlah_foto_uji": len(label_asli),
        "accuracy": float(accuracy_score(label_asli, label_prediksi)),
        "per_kelas": {
            kelas: {
                "precision": float(rincian[kelas]["precision"]),
                "recall": float(rincian[kelas]["recall"]),
                "f1": float(rincian[kelas]["f1-score"]),
                "support": int(rincian[kelas]["support"]),
            }
            for kelas in KELAS
        },
        "macro_avg": {
            "precision": float(rincian["macro avg"]["precision"]),
            "recall": float(rincian["macro avg"]["recall"]),
            "f1": float(rincian["macro avg"]["f1-score"]),
        },
        "confusion_matrix": confusion_matrix(
            label_asli, label_prediksi, labels=indeks_kelas
        ).tolist(),
    }


def simpan_confusion_matrix(
    laporan_publik: dict[str, object] | None,
    laporan_kampus: dict[str, object] | None,
) -> Path:
    """Simpan matriks evaluasi publik dan kampus dengan status yang jujur."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, judul, laporan in (
        (axes[0], "Uji dataset publik", laporan_publik),
        (axes[1], "Uji foto latih kampus", laporan_kampus),
    ):
        ax.set_title(judul)
        if laporan is None:
            ax.text(
                0.5,
                0.5,
                "Belum ada foto uji",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        matriks = np.asarray(laporan["confusion_matrix"])
        gambar = ax.imshow(matriks, cmap="Blues")
        ax.set_xticks(range(len(KELAS)), labels=KELAS, rotation=45, ha="right")
        ax.set_yticks(range(len(KELAS)), labels=KELAS)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label sebenarnya")
        for baris in range(len(KELAS)):
            for kolom in range(len(KELAS)):
                ax.text(kolom, baris, str(matriks[baris, kolom]), ha="center", va="center")
        fig.colorbar(gambar, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    tujuan = REPORT_DIR / "confusion_matrix_latih.png"
    fig.savefig(tujuan, dpi=160)
    plt.close(fig)
    return tujuan


def pilih_device(nama_device: str) -> torch.device:
    if nama_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA diminta, tetapi tidak tersedia pada perangkat ini.")
    if nama_device == "cpu":
        return torch.device("cpu")
    if nama_device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latih(args: argparse.Namespace) -> None:
    atur_seed(args.seed)
    device = pilih_device(args.device)
    print(f"Perangkat pelatihan: {device}")

    path_manifest = args.manifest.resolve()
    baris = baca_manifest(path_manifest, args.max_per_class, args.seed)
    baris_latih = [baris_item for baris_item in baris if baris_item["split"] == "train"]
    baris_val = [baris_item for baris_item in baris if baris_item["split"] == "val"]
    baris_test_publik = [
        item for item in baris if item["split"] == "test" and item["sumber"] != "kampus"
    ]
    baris_test_kampus = [
        item for item in baris if item["split"] == "test" and item["sumber"] == "kampus"
    ]
    if not baris_latih:
        raise ValueError("Manifest tidak memiliki foto pada split train.")
    if not baris_test_publik:
        raise ValueError("Manifest tidak memiliki foto uji dari dataset publik.")

    dilatih_dengan_kampus = any(item["sumber"] == "kampus" for item in baris_latih)
    if not dilatih_dengan_kampus:
        print(
            "PERINGATAN: model tidak dilatih dengan foto kampus; "
            "hasil publik belum mewakili kondisi kampus."
        )

    transformasi_latih, transformasi_evaluasi = buat_transformasi()
    loader_latih = buat_loader(
        baris_latih, transformasi_latih, args.batch_size, True, device, args.seed
    )
    loader_val = (
        buat_loader(baris_val, transformasi_evaluasi, args.batch_size, False, device, args.seed)
        if baris_val
        else None
    )
    loader_test_publik = buat_loader(
        baris_test_publik, transformasi_evaluasi, args.batch_size, False, device, args.seed
    )
    loader_test_kampus = (
        buat_loader(
            baris_test_kampus,
            transformasi_evaluasi,
            args.batch_size,
            False,
            device,
            args.seed,
        )
        if baris_test_kampus
        else None
    )

    bobot = buat_bobot_kelas(baris_latih).to(device)
    fungsi_loss = nn.CrossEntropyLoss(weight=bobot)
    bobot_pretrained = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=bobot_pretrained)
    model.classifier[1] = nn.Linear(model.last_channel, len(KELAS))
    model = model.to(device)

    riwayat = []
    for tahap, laju_belajar in ((1, 1e-3), (2, 1e-4)):
        atur_parameter_model(model, tahap)
        parameter_aktif = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameter_aktif, lr=laju_belajar)
        print(f"\nTahap {tahap}: {args.epochs} epoch, learning rate {laju_belajar}")
        for epoch in range(1, args.epochs + 1):
            loss_latih, akurasi_latih = jalankan_satu_epoch(
                model, loader_latih, fungsi_loss, device, True, optimizer
            )
            hasil_val = None
            if loader_val is not None:
                loss_val, akurasi_val = jalankan_satu_epoch(
                    model, loader_val, fungsi_loss, device, False
                )
                hasil_val = {"loss": loss_val, "accuracy": akurasi_val}
            riwayat.append(
                {
                    "tahap": tahap,
                    "epoch": epoch,
                    "loss_latih": loss_latih,
                    "accuracy_latih": akurasi_latih,
                    "validasi": hasil_val,
                }
            )
            print(
                f"  epoch {epoch}/{args.epochs} "
                f"loss_latih={loss_latih:.4f} accuracy_latih={akurasi_latih:.4f}"
                + (
                    f" loss_val={hasil_val['loss']:.4f} "
                    f"accuracy_val={hasil_val['accuracy']:.4f}"
                    if hasil_val is not None
                    else ""
                )
            )

    label_publik, prediksi_publik = evaluasi(model, loader_test_publik, device)
    laporan_publik = buat_laporan_metrik(label_publik, prediksi_publik)
    if loader_test_kampus is not None:
        label_kampus, prediksi_kampus = evaluasi(model, loader_test_kampus, device)
        laporan_kampus = buat_laporan_metrik(label_kampus, prediksi_kampus)
    else:
        laporan_kampus = None
        print("PERINGATAN: tidak ada foto uji kampus; metrik kampus tidak tersedia.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path_model = MODEL_DIR / "ecolens_mobilenetv2.pt"
    torch.save(
        {
            "state_dict": {nama: tensor.detach().cpu() for nama, tensor in model.state_dict().items()},
            "arsitektur": "MobileNetV2",
            "kelas": list(KELAS),
            "input_size": UKURAN_GAMBAR,
            "normalisasi_mean": list(RATA_RATA_IMAGENET),
            "normalisasi_std": list(STD_IMAGENET),
        },
        path_model,
    )
    path_label = MODEL_DIR / "labels.json"
    with path_label.open("w", encoding="utf-8") as file_json:
        json.dump({str(indeks): kelas for indeks, kelas in enumerate(KELAS)}, file_json, indent=2)

    hasil_laporan = {
        "seed": args.seed,
        "arsitektur": "MobileNetV2",
        "bobot_awal": str(bobot_pretrained),
        "perangkat": str(device),
        "model_dilatih_dengan_foto_kampus": dilatih_dengan_kampus,
        "jumlah_foto_latih_per_sumber": dict(
            Counter(item["sumber"] for item in baris_latih)
        ),
        "riwayat_latih": riwayat,
        "evaluasi_dataset_publik": laporan_publik,
        "evaluasi_foto_latih_kampus": laporan_kampus,
        "catatan": (
            "Evaluasi publik dan kampus dipisahkan. Foto kampus tanpa label manual "
            "tidak digunakan untuk metrik kampus."
        ),
    }
    path_metrics = REPORT_DIR / "metrics_latih.json"
    with path_metrics.open("w", encoding="utf-8") as file_json:
        json.dump(hasil_laporan, file_json, indent=2, ensure_ascii=False)
    path_matriks = simpan_confusion_matrix(laporan_publik, laporan_kampus)

    print("\nFile hasil:")
    print(f"  Model: {path_model}")
    print(f"  Label: {path_label}")
    print(f"  Metrik: {path_metrics}")
    print(f"  Confusion matrix: {path_matriks}")
    print(f"Accuracy uji publik: {laporan_publik['accuracy']:.4f}")
    if laporan_kampus is None:
        print("Metrik uji foto kampus: tidak tersedia (belum ada foto audit berlabel).")
    else:
        print(f"Accuracy uji foto kampus: {laporan_kampus['accuracy']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROCESSED_DIR / "manifest_latih.csv",
        help="Lokasi manifest hasil persiapan data.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epoch pada setiap tahap.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Batas foto per kelas/sumber/split; 0 berarti tidak dibatasi.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Perangkat; auto memilih CUDA bila tersedia, selain itu CPU.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Seed reproducible.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs harus minimal 1.")
    if args.max_per_class < 0:
        parser.error("--max-per-class tidak boleh negatif.")
    if args.batch_size < 1:
        parser.error("--batch-size harus minimal 1.")

    try:
        latih(args)
    except (FileNotFoundError, ValueError) as kesalahan:
        print(f"ERROR: {kesalahan}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
