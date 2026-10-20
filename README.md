# SIMPEMI - Simulasi Rekomendasi Peminatan dan Mata Kuliah Pilihan 🎓

**SIMPEMI** adalah aplikasi web sistem pendukung keputusan (Decision Support System) berbasis Machine Learning untuk membantu mahasiswa Program Studi Informatika dalam menentukan jalur peminatan (**Kecerdasan Buatan (AI)**, **Sains Data / Data Science**, atau **Rekayasa Perangkat Lunak (RPL)**) serta rekomendasi mata kuliah pilihan terkait, dilengkapi fitur simulasi interaktif skenario *What-If*.

Project ini dikembangkan untuk keperluan **Capstone Design Informatika / Tugas Akhir**.

---

## 🚀 Fitur Utama & Pembaruan

1. **🔄 Dukungan Multi-Semester Kumulatif (Semester 4 s/d 7):**
   - **Semester 4 (2024/2025):** Mengisi nilai 9 mata kuliah Semester 4 dan IPS Semester 4.
   - **Semester 5 (2025/2026):** Mengisi nilai mata kuliah Semester 4 & 5 serta IPS Semester 4 & 5.
   - **Semester 6 (2025/2026):** Mengisi nilai mata kuliah Semester 4, 5 & 6 serta IPS Semester 4, 5 & 6.
   - **Semester 7 (2026/2027):** Mengisi nilai mata kuliah Semester 4, 5, 6 & 7 serta IPS Semester 4, 5, 6 & 7.

2. **📈 Penggantian IPK Menjadi IPS per Semester:**
   - Input IPK statis digantikan dengan input **IPS (Indeks Prestasi Semester)** individual untuk tiap semester yang telah ditempuh (skala 0.00 – 4.00).
   - Estimasi IPK kumulatif dihitung secara dinamis dan otomatis oleh sistem.

3. **🎯 7 Domain Minat Bidang Informatika Terintegrasi:**
   - 1. Kecerdasan Buatan (AI)
   - 2. Sains Data (Data Science)
   - 3. Software & Web (Rekayasa Perangkat Lunak)
   - 4. Keamanan Siber & Jaringan Komputer (Cybersecurity & Networking)
   - 5. Sistem Terdistribusi & Komputasi Paralel (Distributed Systems & Parallel Computing)
   - 6. Teknologi Game & Multimedia (Game Development & Computer Graphics)
   - 7. Internet of Things (IoT) & Sistem Embedded

4. **🛠️ 6 Keterampilan Teknis Terapan:**
   - 1. Skill Python (AI, Machine Learning, Data Analytics)
   - 2. Skill SQL (Basis Data Relasional & Manipulasi Data)
   - 3. Skill Web & Mobile Dev (Frontend & Backend Frameworks)
   - 4. Skill Pemrograman OOP (Java/C++/C# & Design Patterns)
   - 5. Skill Jaringan Komputer & Linux CLI (Routing, IP, Server Admin)
   - 6. Skill Keamanan Siber (Security Tools, Wireshark, Pen-testing)

5. **🤖 Model Klasifikasi Multi-Tier:**
   - Melatih 4 tingkatan model terpisah untuk masing-masing semester (Semester 4, 5, 6, 7).
   - Menggunakan algoritma **Random Forest Classifier** (akurasi 82% – 84%), dibandingkan dengan **Logistic Regression**, **KNN**, dan **Decision Tree**.

5. **👤 Identitas Mahasiswa & Pencatatan Otomatis (Database Riwayat):**
   - Mahasiswa wajib mengisi **Nama Lengkap**, **NIM**, dan **Kelas** sebelum melakukan simulasi.
   - Hasil rekomendasi dan profil nilai mahasiswa otomatis tercatat secara persisten ke dalam berkas `data_riwayat_mahasiswa.csv`.

6. **🎯 Simulasi Rekomendasi & Radar Chart Kompetensi:**
   - Opsi pengisian cepat dengan profil preset (*Condong AI*, *Condong Data Science*, atau *Condong RPL*).
   - Peringatan cerdas (*warning alert*) jika selisih probabilitas peringkat 1 dan 2 sangat tipis (< 10%).
   - Visualisasi Radar Chart indeks kompetensi mahasiswa vs profil rata-rata kelompok peminatan.
   - Analisis *Feature Importance* untuk mengetahui mata kuliah semester tersebut yang paling menentukan rekomendasi.
   - **🧪 Mode Simulasi What-If:** Menguji skenario kenaikan/penurunan nilai mata kuliah kunci atau keterampilan teknis dan melihat pergeseran probabilitas secara real-time.
   - Ekspor data profil dan hasil rekomendasi ke format CSV.

7. **📊 Evaluasi Model per Semester:**
   - Komparasi 4 algoritma klasifikasi lengkap dengan *Accuracy*, *Precision*, *Recall*, *F1-Score*.
   - Visualisasi *Confusion Matrix* interaktif dan grafik *Feature Importance*.

8. **🗂️ Dataset & Manajemen Riwayat Mahasiswa (Dual-Tab):**
   - **Tab 1 (📋 Data Riwayat Mahasiswa):** Menampilkan daftar mahasiswa yang telah melakukan simulasi, fitur pencarian instan (Nama/NIM/Kelas), ekspor CSV riwayat, dan opsi reset.
   - **Tab 2 (🧠 Dataset Latih AI):** 800 sampel data acuan pelatihan model machine learning lengkap dengan statistik deskriptif dan histogram distribusi fitur.

---

## 📚 Kurikulum Mata Kuliah per Semester

### Semester 4 (2024/2025)
- Komunikasi Interpersonal
- Matematika Informatika 3 (Logika Mat) 2SKS
- Pemrograman Berorientasi Objek
- Pengantar Bahasa Dan Automata
- Pengantar Interaksi Manusia Dan Komputer
- Pengantar Metode Numerik
- Pengantar Rangkaian Digital
- Prak Pemrograman Berorientasi Objek
- Rekayasa Perangkat Lunak
- *IPS Semester 4*

### Semester 5 (2025/2026)
- Etika Profesi Informatika
- Interface & Periferal
- Keamanan Jaringan
- Mikroprosessor & Mikrokontroler
- Pemrograman Multimedia
- Pengolahan Video
- Prak Pemrograman Multimedia
- RPL Berbasis Objek - Augmented Reality
- Sistem Multimedia
- Sistem Operasi
- Teknik Kompilasi
- *IPS Semester 5*

### Semester 6 (2025/2026)
- Keamanan Cyber
- Kecerdasan Buatan
- Kerja Praktek / Penulisan Ilmiah
- Pemrograman Web
- Pengantar Komputer Vision
- Pengantar Sistem Informasi Manajemen
- Prak Pemrograman Web
- Probabilitas & Statistika
- Riset Teknik Informasi / Metode Penelitian
- Teknik Data Mining
- Wireless / Mobile Computing
- *IPS Semester 6*

### Semester 7 (2026/2027)
- Praktikum Data Mining
- Praktikum Jaringan
- Komputer Paralel (Sistem Terdistribusi)
- Teknik Simulasi dan Pemodelan
- Pengolahan Citra
- Pemrograman Game
- Jaringan Syaraf Tiruan
- Pemrograman Jaringan
- Manajemen Proyek (Capstone Project)
- Machine Learning
- Jaringan Komputer dan Komunikasi Data
- *IPS Semester 7*

---

## 🛠️ Prasyarat dan Instalasi

### 1. Prasyarat Sistem
- **Python:** Versi 3.10 atau lebih baru (disarankan 3.11 atau 3.12).
- Terminal / PowerShell / Command Prompt.

### 2. Langkah Instalasi

1. **Buka folder project:**
   ```bash
   cd "C:\Users\Adam\Downloads\Tugas PRoject CAPSTONE"
   ```

2. **Aktifkan Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

3. **Install dependensi library (jika belum):**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Cara Menjalankan Aplikasi

Jalankan perintah berikut di terminal:

```bash
streamlit run app.py
```
*Atau:*
```bash
python -m streamlit run app.py
```

Setelah perintah dijalankan, buka browser di alamat:
👉 **`http://localhost:8501`**
