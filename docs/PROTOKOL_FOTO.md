# Protokol Foto Lapangan

## Aturan pengambilan foto

- Ambil foto hanya pada titik kampus yang sudah disepakati tim dan dicatat dengan kode lokasi.
- Ambil foto pada pagi (05:00-10:59), siang (11:00-14:59), atau sore (15:00-18:59).
- Satu foto memuat satu objek sampah atau satu tumpukan yang sejenis.
- Ambil foto dari jarak dan sudut yang cukup agar jenis objek terlihat; hindari foto buram.
- Jangan memasukkan wajah, plat nomor, nama, layar ponsel, atau informasi pribadi lain ke foto.
- Jangan membagikan foto yang memuat informasi pribadi. Tinjau foto sebelum menyalinnya ke folder proyek.
- Hapus metadata EXIF, termasuk GPS, sebelum foto diproses atau dibagikan.
- Catat foto audit dengan pola `KODELOKASI_YYYYMMDD_HHMM_NOMOR.jpg`.
- Satu objek fisik boleh memiliki paling banyak tiga foto latih. Gunakan ID objek yang sama pada nama semua fotonya.
- Foto audit tidak boleh dipakai untuk melatih model.
- Satuan analisis adalah jumlah foto/item, bukan berat atau volume. Beberapa foto dari satu objek dapat memengaruhi hitungan.

## Panduan label

- `plastik`: objek atau kemasan yang bahan utamanya plastik.
- `kertas_kardus`: objek yang bahan utamanya kertas atau kardus.
- `logam`: objek yang bahan utamanya logam.
- `kaca`: objek yang bahan utamanya kaca.
- `organik`: sisa bahan hayati, misalnya sisa makanan atau daun.
- `residu`: objek yang tidak dapat dipastikan masuk kelas lain dalam panduan ini.
- Pilih satu label utama untuk satu objek atau satu tumpukan sejenis. Jika ragu, catat untuk verifikasi tim dan jangan menebak.
- Label audit hanya diisi untuk foto yang dipilih tim untuk evaluasi; catat pasangan nama file dan label di `data/raw/label_audit.csv`.

## Keselamatan

- Jangan menyentuh, membuka, atau memindahkan sampah untuk mengambil foto.
- Jangan mengambil foto dari jalan aktif, area terlarang, atau posisi yang berisiko jatuh/terluka.
- Hindari benda tajam, limbah medis, cairan tak dikenal, dan bahan berbahaya; jaga jarak dan laporkan kepada petugas kampus.
- Ikuti aturan kampus dan minta izin bila lokasi atau kegiatan memerlukannya.
