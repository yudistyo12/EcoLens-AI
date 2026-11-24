# Panduan Deployment Dashboard

Panduan ini menyiapkan dashboard agar dapat dibuka publik tanpa login. Tim
menerbitkan repo dari akun GitHub milik sendiri; belum ada URL publik sampai
deployment berhasil dan diuji.

## Sebelum deploy

- Jalankan `python -m src.analyze_public` dan `python -m src.export_onnx`.
- Periksa `reports/checklist_pengumpulan.json`; jangan klaim proyek siap bila
  status masih `BELUM SIAP`.
- Tinjau semua CSV/laporan yang akan dimasukkan repo. Pastikan tidak ada nama
  orang, wajah, plat nomor, koordinat pribadi, atau informasi sensitif.
- Jangan unggah seluruh dataset Kaggle atau foto pribadi. Pastikan kebijakan
  lisensi sumber dipatuhi.
- Repo perlu memuat kode, manifest sampel, laporan hasil, berkas model ONNX,
  `models/labels.json`, dan dependensi dashboard. Pastikan ukuran berkas sesuai
  batas platform dan GitHub.
- Penyimpanan foto/riwayat server hanya aktif untuk akses localhost. Pada
  dashboard publik, pengguna dapat mengunduh hasil sebagai CSV.

## Menerbitkan di Streamlit Community Cloud

1. Buat repo GitHub publik pada akun tim dan tinjau file yang akan dipublikasikan.
   Jangan unggah token, rahasia, foto mentah, atau data pribadi.
2. Dorong kode proyek ke repo dengan prosedur Git tim.
3. Buka Streamlit Community Cloud dan hubungkan akun GitHub.
4. Buat aplikasi baru dengan repo tersebut, branch yang berisi kode, dan file
   utama `app.py`.
5. Pastikan `requirements.txt` berada di akar repo dan model ONNX MobileNetV2,
   model pemeriksaan wajah YuNet, lisensinya, serta laporan yang dibutuhkan ikut
   tersedia.
6. Tunggu proses build. Jika gagal, baca log build dan perbaiki sebelum dibagikan.
7. Buka URL yang diberikan pada jendela privat/incognito tanpa login. Periksa
   sumber data, filter, halaman metrik, halaman rekomendasi, scan kamera, dan
   unggah foto. Kamera memerlukan izin browser dan akses HTTPS; gunakan unggah
   file bila perangkat menolak akses kamera. Pastikan deteksi wajah menolak foto
   berwajah dan konfirmasi foto wajib muncul sebelum hasil klasifikasi.
8. Catat URL yang benar-benar berhasil diuji dan isi di bahan pengumpulan.

Deployment belum dilakukan oleh skrip lokal ini. URL publik hanya boleh
dinyatakan tersedia setelah langkah pengujian akses tanpa login berhasil.
