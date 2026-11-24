# Rekomendasi Tim untuk Demo Klasifikasi Sampah

Usulan berikut disusun dari evaluasi split uji dataset publik Kaggle. Ini
rekomendasi untuk pengembangan model, bukan kesimpulan tentang pola sampah atau
kebutuhan fasilitas kampus.

## Rekomendasi 1

- Usulan tim: Tinjau kesalahan kelas plastik, lalu tambahkan contoh berlisensi
  dan bervariasi ke data latih jika audit menunjukkan variasi gambar belum
  terwakili. Periksa label sebelum melatih ulang.
- Insight/angka hasil eksekusi yang mendasari: Recall plastik terendah, yaitu
  75,6% (34 dari 45 foto uji kelas plastik dikenali sebagai plastik).
- Dampak yang diharapkan untuk klasifikasi atau edukasi pemilahan: Mengurangi
  foto plastik yang terlewat dan membantu tim menjelaskan bahwa hasil klasifikasi
  masih perlu ditinjau. Dampak ini harus dibuktikan lewat evaluasi ulang.
- Cara menguji usulan: Audit contoh yang salah diklasifikasikan tanpa
  memindahkannya ke data latih; perbaiki data latih/panduan label berdasarkan
  audit; lalu bandingkan recall per kelas, macro F1, dan confusion matrix pada
  split uji terpisah yang tidak dipakai untuk penyetelan.
- Keterbatasan: Hanya ada 45 foto uji per kelas pada evaluasi ini. Temuan berasal
  dari dataset publik dan belum mengukur performa pada foto kampus atau kondisi
  kamera nyata.

## Rekomendasi 2

- Usulan tim: Perjelas panduan anotasi untuk membedakan kaca dan logam dengan
  contoh visual berlisensi; minta peninjau memeriksa label yang ambigu sebelum
  data digunakan untuk pelatihan.
- Insight/angka hasil eksekusi yang mendasari: Kesalahan tunggal yang paling
  sering muncul adalah 4 foto berlabel kaca diprediksi sebagai logam.
- Dampak yang diharapkan untuk klasifikasi atau edukasi pemilahan: Mengurangi
  kebingungan label dan membuat penjelasan perbedaan kedua kelas lebih konsisten.
  Perbaikan performa model belum dapat diklaim sebelum diuji.
- Cara menguji usulan: Tinjau empat contoh kesalahan, tetapkan panduan label,
  lalu uji model hasil pembaruan pada foto berlabel manual yang terpisah dari
  data pelatihan. Laporkan confusion matrix serta recall dan F1 kelas kaca dan
  logam sebelum dan sesudah.
- Keterbatasan: Empat kesalahan saja belum menjelaskan penyebab tertukarnya
  kelas. Foto Kaggle juga belum tentu mewakili variasi kaca dan logam di kampus.

## Pemeriksaan sebelum dipresentasikan

- Angka yang dicantumkan berasal dari `reports/analisis_kaggle.json`.
- Bedakan hasil uji Kaggle dari performa pada foto kampus.
- Jelaskan bahwa label dan distribusi dataset publik tidak otomatis mewakili sampah lokal.
