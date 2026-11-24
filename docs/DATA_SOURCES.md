# Sumber Data EcoLens AI

## Dataset utama

- Nama: Garbage Dataset (Garbage Classification V2).
- Pembuat: Suman Kunwar.
- Sumber: Kaggle, `sumn2u/garbage-classification-v2`
  (https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2).
- Lisensi yang dicantumkan sumber: MIT; informasi lisensi dicatat diperiksa
  24 November 2026.
- Versi sumber: 12. Metadata dataset menyebut 10 kelas dan 13.348 foto.
- Folder yang digunakan: hanya `standardized_256`. Folder `original` dan
  `standardized_384` tidak digunakan karena berisi foto yang sama.
- Kelas yang dipakai: `biological`, `plastic`, `paper`, `cardboard`, `metal`,
  `glass`, dan `trash`. Folder `battery`, `shoes`, dan `clothes` diabaikan.
- Pemetaan: `biological` ke `organik`; `plastic` ke `plastik`; `paper` dan
  `cardboard` ke `kertas_kardus`; `metal` ke `logam`; `glass` ke `kaca`; dan
  `trash` ke `residu`.
- Jumlah file gambar yang ditemukan pada folder lokal, setelah pemetaan, serta
  jumlah sampel manifest dihitung saat eksekusi dan disimpan pada
  `reports\analisis_kaggle.json` serta `reports\inventaris_gd_v2.csv`.
- Peran: pelatihan, evaluasi, dan analisis klasifikasi sampah; dataset ini tidak
  memiliki metadata lokasi/waktu Universitas Teknokrat Indonesia.
- Rujukan: Kunwar, S. (2026). *J. Braz. Comput. Soc.*, 32(1).
  https://doi.org/10.5753/jbcs.2026.7774

## Data yang tidak digunakan

TrashNet dan foto lapangan kampus tidak digunakan pada demo ini. Foto Kaggle
tidak diperlakukan sebagai foto audit kampus. Tidak ada scraping gambar dari
Google atau sumber tanpa lisensi yang jelas.

## Model pemeriksaan wajah

- Model: YuNet `face_detection_yunet_2023mar.onnx`.
- Sumber: OpenCV Zoo,
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet.
- Lisensi: MIT; teks lisensi disertakan di `models/LICENSE-YUNET.txt`.
- Fungsi: pemeriksaan awal untuk menolak foto yang wajahnya terdeteksi sebelum
  klasifikasi sampah. Model tidak mengenali identitas seseorang.
- Batas: deteksi dapat melewatkan wajah kecil, tertutup, menyamping, atau gelap,
  dan dapat salah menandai bentuk tertentu. Tidak mendeteksi semua jenis foto
  non-sampah; pengguna tetap harus memastikan isi foto.

## Batas interpretasi

- Metrik menjelaskan performa pada split dataset publik, bukan kondisi kampus.
- Manifest dibatasi per kelas dan digunakan untuk evaluasi/pelatihan model;
  distribusinya tidak menunjukkan prevalensi jenis sampah di lingkungan nyata.
- Data sumber kelas `paper` dan `cardboard` digabung menjadi
  `kertas_kardus`.
- Kelas dan pemetaan harus dijelaskan saat presentasi, termasuk keterbatasan
  dataset dan kemungkinan perbedaan kondisi foto dunia nyata.
