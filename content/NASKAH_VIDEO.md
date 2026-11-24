# Naskah Video Demo EcoLens AI

Durasi yang disarankan: 2-3 menit. Ini adalah naskah, bukan bukti bahwa video
telah direkam atau diunggah. Tim perlu merekam dashboard yang benar-benar
berjalan, mencantumkan URL publik aktif, dan membagikan tautan video.

## Alur rekaman

1. **Pembuka (0:00-0:20)**
   - "Kami memperkenalkan EcoLens AI, demo klasifikasi sampah menggunakan
     computer vision."
   - "Proyek ini memakai dataset publik Garbage Classification V2 dari Kaggle,
     bukan foto audit kampus."

2. **Data dan metode (0:20-0:55)**
   - Buka halaman Ikhtisar data. Tunjukkan sumber, jumlah foto lokal dan
     manifest yang dihitung oleh aplikasi.
   - Jelaskan bahwa subset per kelas digunakan untuk pelatihan dan tidak
     mewakili prevalensi sampah di dunia nyata.
   - Jelaskan MobileNetV2 dan pemetaan enam label.

3. **Evaluasi (0:55-1:35)**
   - Buka halaman Evaluasi model. Tunjukkan accuracy, F1 per kelas, confusion
     matrix, dan kurva pelatihan.
   - Sebutkan bahwa evaluasi berasal dari split uji Kaggle; jangan menyebutnya
     akurasi di kampus.

4. **Inferensi dan rekomendasi (1:35-2:20)**
   - Unggah satu foto sampah tanpa wajah, plat nomor, atau informasi pribadi.
   - Jelaskan prediksi, confidence, dan bahwa foto tidak disimpan.
   - Tunjukkan kandidat rekomendasi. Jelaskan bahwa tim meninjau kandidat dan
     tidak membuat rekomendasi fasilitas kampus dari data ini.

5. **Penutup (2:20-2:40)**
   - Sebutkan sumber dataset, keterbatasan, dan langkah lanjutan untuk menguji
     model dengan foto yang dikumpulkan secara etis.
   - Tampilkan nama anggota tim dan tautan dashboard yang benar-benar dapat
     dibuka tanpa login.

## Sebelum mengunggah

- Pastikan semua anggota memahami dan dapat menjelaskan kode serta metrik.
- Jangan menyatakan URL publik aktif sebelum diuji di jendela privat.
- Lengkapi disclosure AI dan pernyataan orisinalitas sesuai aturan lomba.
