# EcoLens AI

Dashboard klasifikasi sampah berbasis computer vision untuk mendukung edukasi
pemilahan. Dataset utama adalah Garbage Classification V2 dari Kaggle.
Proyek ini tidak memetakan lokasi/jam Universitas Teknokrat Indonesia dan tidak
menghasilkan rekomendasi penempatan fasilitas kampus.

## Data dan batas klaim

- Dataset publik dipakai untuk pelatihan, evaluasi, dan analisis model.
- Sumber, lisensi, pemetaan kelas, jumlah file lokal, dan keterbatasan ada di
  `docs\DATA_SOURCES.md` dan `reports\analisis_kaggle.json`.
- Jumlah sampel pada manifest dibatasi dan diseimbangkan untuk pelatihan; jangan
  mengartikannya sebagai prevalensi sampah di dunia nyata.
- Metrik dari split uji Kaggle tidak sama dengan akurasi pada foto kampus.
- Unggah satu foto pada dashboard hanya untuk inferensi sesi; file tidak disimpan.

## Lokasi dataset

Letakkan folder `standardized_256` pada
`data\public\garbage_v2\standardized_256`. Skrip juga mendukung lokasi dataset
yang ada di workspace ini, `dataset\standardized_256`. Folder lain, termasuk
`original` dan `standardized_384`, tidak dipakai.

## Menyiapkan, melatih, dan menganalisis model

Jalankan dari akar proyek dengan Python 3.10+:

```powershell
python -m pip install -r requirements-train.txt
python -m src.prepare_data --max-publik 300
python -m src.train --epochs 5 --max-per-class 300 --batch-size 32 --device cpu
python -m src.analyze_public
python -m src.export_onnx
python -m src.benchmark_onnx
python -m src.create_summary
```

Manifest membagi foto dengan seed tetap. Skrip analisis menghasilkan insight
terhitung, inventaris dataset, dan kandidat rekomendasi berbasis kesalahan model.
Kandidat bukan rekomendasi final; tim wajib meninjau dan menuliskan keputusan
akhir sendiri. Benchmark ONNX memakai foto split uji dan hanya menghitung waktu
pemanggilan inferensi; preprocessing tidak termasuk. Ringkasan memverifikasi
hash benchmark terhadap model dan manifest saat PDF dibuat.

Pemeriksaan submission juga memeriksa file gambar yang hilang, duplikasi isi
gambar melalui SHA-256, dan `group_id` yang tersebar di beberapa split.

## Menjalankan dashboard

Pasang dependensi dashboard dari `requirements.txt`, lalu:

```powershell
streamlit run app.py
```

Dashboard berisi analisis sumber data, metrik per kelas, confusion matrix,
riwayat pelatihan, filter, serta klasifikasi satu foto melalui kamera atau
unggahan file. Hasil menampilkan skor softmax dan waktu inferensi; halaman model
menampilkan ukuran berkas ONNX dan rata-rata waktu inferensi pada foto yang
dicoba dalam sesi aktif. Kamera memerlukan izin browser dan HTTPS atau localhost;
unggahan file tetap tersedia sebagai alternatif. Hasil bisa diunduh sebagai CSV.
Logo EcoLens AI tersedia sebagai berkas vektor di `assets\ecolens-logo.svg`;
ikon mandirinya ada di `assets\ecolens-mark.svg`.
Foto yang terdeteksi memuat wajah ditolak sebelum klasifikasi. Pemeriksaan ini
tidak menjamin bahwa foto tanpa wajah adalah foto sampah; pengguna tetap harus
memastikan isi foto dan tidak mengunggah wajah, plat nomor, atau data pribadi.
Model pemeriksaan wajah YuNet dan lisensinya harus ikut tersedia di folder
`models`.
Penyimpanan foto/riwayat lokal otomatis tersedia saat aplikasi dibuka di
localhost dan tidak aktif di dashboard publik.

Jalankan dashboard secara lokal dengan:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1
```

Tombol simpan menyimpan hasil dan salinan foto JPEG yang telah diubah ukurannya
ke `data\local_history`. Folder tersebut diabaikan Git. Riwayat dapat diunduh
atau dihapus dari halaman klasifikasi.

## Pemeriksaan dan pengumpulan

```powershell
python -m src.check_submission
```

Laporan pemeriksaan tersimpan di `reports\checklist_pengumpulan.json`. Panduan
deployment tersedia di `docs\DEPLOYMENT.md`. Ringkasan PDF otomatis dibuat di
`submission\GreenPixel_Universitas Teknokrat Indonesia.pdf`; nama tim, ketua,
anggota, dan kampus sudah dicantumkan. Naskah video presentasi
ada di `content\NASKAH_VIDEO.md`. URL dashboard publik, video yang sudah direkam,
dan surat pernyataan orisinalitas tetap perlu disiapkan/diperiksa oleh tim.
