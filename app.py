"""
SIMPEMI - Simulasi Rekomendasi Peminatan dan Mata Kuliah Pilihan
Capstone Design Informatika | AI - Rekomendasi
Mendukung Mahasiswa Semester 4 hingga 7 dengan Pengisian Kumulatif, IPS per Semester,
7 Domain Minat Bidang, dan 6 Keterampilan Teknis Terintegrasi.
Jalankan: streamlit run app.py
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="SIMPEMI - Rekomendasi Peminatan", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# LOG PERSISTENSI DATA MAHASISWA (RIWAYAT SIMULASI)
# ----------------------------------------------------------------------------
LOG_MAHASISWA_FILE = "data_riwayat_mahasiswa.csv"

def muat_riwayat_mahasiswa():
    """Memuat data riwayat mahasiswa yang telah melakukan simulasi."""
    if os.path.exists(LOG_MAHASISWA_FILE):
        try:
            return pd.read_csv(LOG_MAHASISWA_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def simpan_riwayat_mahasiswa(entry: dict):
    """Menyimpan data hasil simulasi mahasiswa ke file CSV secara persisten."""
    df_new = pd.DataFrame([entry])
    if os.path.exists(LOG_MAHASISWA_FILE):
        try:
            df_existing = pd.read_csv(LOG_MAHASISWA_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(LOG_MAHASISWA_FILE, index=False)
        except Exception:
            df_new.to_csv(LOG_MAHASISWA_FILE, index=False)
    else:
        df_new.to_csv(LOG_MAHASISWA_FILE, index=False)

def hapus_riwayat_mahasiswa():
    """Menghapus seluruh file riwayat data mahasiswa."""
    if os.path.exists(LOG_MAHASISWA_FILE):
        os.remove(LOG_MAHASISWA_FILE)

# ----------------------------------------------------------------------------
# KONFIGURASI KURIKULUM & FITUR
# ----------------------------------------------------------------------------
KELAS = ["AI", "Data Science", "RPL"]

TA_SEMESTER = {
    4: "2024/2025",
    5: "2025/2026",
    6: "2025/2026",
    7: "2026/2027",
}

MATKUL_PER_SEMESTER = {
    4: [
        "Komunikasi Interpersonal",
        "Matematika Informatika 3 (Logika Mat)",
        "Pemrograman Berorientasi Objek",
        "Pengantar Bahasa Dan Automata",
        "Pengantar Interaksi Manusia Dan Komputer",
        "Pengantar Metode Numerik",
        "Pengantar Rangkaian Digital",
        "Prak Pemrograman Berorientasi Objek",
        "Rekayasa Perangkat Lunak",
    ],
    5: [
        "Etika Profesi Informatika",
        "Interface & Periferal",
        "Keamanan Jaringan",
        "Mikroprosessor & Mikrokontroler",
        "Pemrograman Multimedia",
        "Pengolahan Video",
        "Prak Pemrograman Multimedia",
        "RPL Berbasis Objek - Augmented Reality",
        "Sistem Multimedia",
        "Sistem Operasi",
        "Teknik Kompilasi",
    ],
    6: [
        "Keamanan Cyber",
        "Kecerdasan Buatan",
        "Kerja Praktek / Penulisan Ilmiah",
        "Pemrograman Web",
        "Pengantar Komputer Vision",
        "Pengantar Sistem Informasi Manajemen",
        "Prak Pemrograman Web",
        "Probabilitas & Statistika",
        "Riset Teknik Informasi / Metode Penelitian",
        "Teknik Data Mining",
        "Wireless / Mobile Computing",
    ],
    7: [
        "Praktikum Data Mining",
        "Praktikum Jaringan",
        "Komputer Paralel (Sistem Terdistribusi)",
        "Teknik Simulasi dan Pemodelan",
        "Pengolahan Citra",
        "Pemrograman Game",
        "Jaringan Syaraf Tiruan",
        "Pemrograman Jaringan",
        "Manajemen Proyek (Capstone Project)",
        "Machine Learning",
        "Jaringan Komputer dan Komunikasi Data",
    ],
}

# 7 Domain Minat Bidang Informatika
MINAT_COLS = [
    "Minat AI",
    "Minat Data",
    "Minat Software/Web",
    "Minat Keamanan Siber & Jaringan",
    "Minat Sistem Terdistribusi & Komputasi Paralel",
    "Minat Teknologi Game & Multimedia",
    "Minat IoT & Sistem Embedded",
]

# 6 Keterampilan Teknis Terapan
SKILL_COLS = [
    "Skill Python",
    "Skill SQL",
    "Skill Web Dev",
    "Skill OOP (Java/C++)",
    "Skill Jaringan & Linux",
    "Skill Keamanan Siber",
]

# Daftar mata kuliah pilihan / peminatan lanjutan yang disarankan per jalur
MATKUL = {
    "AI": [
        "Machine Learning Lanjut",
        "Deep Learning & Generative AI",
        "Natural Language Processing (NLP)",
        "Computer Vision & Pengenalan Objek",
        "Jaringan Syaraf Tiruan (JST)",
        "Sistem Robotika & Otomasi Cerdas",
    ],
    "Data Science": [
        "Data Mining & Warehouse",
        "Big Data Analytics & Cloud Processing",
        "Visualisasi Data & Dashboard Interaktif",
        "Analitika Prediktif & Business Intelligence",
        "Statistika Komputasi & Inferensi Lanjut",
        "Teknik Simulasi & Pemodelan Data",
    ],
    "RPL": [
        "Arsitektur & Pola Desain Perangkat Lunak",
        "Pengujian & Penjaminan Kualitas Software (QA)",
        "DevOps & CI/CD Pipeline Automation",
        "Pemrograman Web & Mobile Enterprise",
        "Microservices & Cloud Computing",
        "Pengembangan Game Berbasis Mesin Grafis",
    ],
}

HELP = {
    # Semester 4
    "Komunikasi Interpersonal": "Nilai akhir (0-100) Komunikasi Interpersonal (KHS).",
    "Matematika Informatika 3 (Logika Mat)": "Nilai akhir (0-100) Matematika Informatika 3 / Logika Matematika 2 SKS.",
    "Pemrograman Berorientasi Objek": "Nilai akhir (0-100) Teori Pemrograman Berorientasi Objek (PBO).",
    "Pengantar Bahasa Dan Automata": "Nilai akhir (0-100) Pengantar Teori Bahasa & Otomata.",
    "Pengantar Interaksi Manusia Dan Komputer": "Nilai akhir (0-100) Interaksi Manusia & Komputer (IMK / UI-UX).",
    "Pengantar Metode Numerik": "Nilai akhir (0-100) Pengantar Metode Numerik & komputasi numerik.",
    "Pengantar Rangkaian Digital": "Nilai akhir (0-100) Pengantar Rangkaian & Logika Digital.",
    "Prak Pemrograman Berorientasi Objek": "Nilai praktikum (0-100) Praktikum Pemrograman Berorientasi Objek.",
    "Rekayasa Perangkat Lunak": "Nilai akhir (0-100) Rekayasa Perangkat Lunak (RPL / SDLC).",

    # Semester 5
    "Etika Profesi Informatika": "Nilai akhir (0-100) Etika Profesi Informatika & Cyber Law.",
    "Interface & Periferal": "Nilai akhir (0-100) Interface & Komunikasi Periferal Hardware.",
    "Keamanan Jaringan": "Nilai akhir (0-100) Keamanan Jaringan & Protokol Keamanan.",
    "Mikroprosessor & Mikrokontroler": "Nilai akhir (0-100) Sistem Mikroprosesor & Mikrokontroler.",
    "Pemrograman Multimedia": "Nilai akhir (0-100) Teori Pemrograman Grafis & Multimedia.",
    "Pengolahan Video": "Nilai akhir (0-100) Pengolahan Video Digital & Komposisi.",
    "Prak Pemrograman Multimedia": "Nilai praktikum (0-100) Praktikum Pemrograman Multimedia.",
    "RPL Berbasis Objek - Augmented Reality": "Nilai akhir (0-100) RPL Berbasis Objek & Augmented Reality (AR).",
    "Sistem Multimedia": "Nilai akhir (0-100) Konsep Sistem Multimedia Terpadu.",
    "Sistem Operasi": "Nilai akhir (0-100) Sistem Operasi (Manajemen Proses, Memori, Storage).",
    "Teknik Kompilasi": "Nilai akhir (0-100) Teknik Kompilasi, Lexer, Parser, & Generator Kode.",

    # Semester 6
    "Keamanan Cyber": "Nilai akhir (0-100) Keamanan Siber (Penetration Testing, Kriptografi, Forensik).",
    "Kecerdasan Buatan": "Nilai akhir (0-100) Pengantar Kecerdasan Buatan & Sistem Cerdas.",
    "Kerja Praktek / Penulisan Ilmiah": "Nilai akhir (0-100) Kerja Praktek / Penulisan Ilmiah.",
    "Pemrograman Web": "Nilai akhir (0-100) Pemrograman Web Fullstack.",
    "Pengantar Komputer Vision": "Nilai akhir (0-100) Pengantar Computer Vision & Ekstraksi Fitur Citra.",
    "Pengantar Sistem Informasi Manajemen": "Nilai akhir (0-100) Pengantar Sistem Informasi Manajemen (SIM).",
    "Prak Pemrograman Web": "Nilai praktikum (0-100) Praktikum Pemrograman Web Fullstack.",
    "Probabilitas & Statistika": "Nilai akhir (0-100) Probabilitas & Statistika Terapan.",
    "Riset Teknik Informasi / Metode Penelitian": "Nilai akhir (0-100) Metodologi Penelitian Bidang Informatika.",
    "Teknik Data Mining": "Nilai akhir (0-100) Teknik Data Mining (Klasifikasi, Asosiasi, Clustering).",
    "Wireless / Mobile Computing": "Nilai akhir (0-100) Wireless & Mobile Computing / Pengembangan Aplikasi Mobile.",

    # Semester 7
    "Praktikum Data Mining": "Nilai praktikum (0-100) Praktikum Ekplorasi & Pemodelan Data Mining.",
    "Praktikum Jaringan": "Nilai praktikum (0-100) Praktikum Administrasi & Routing Jaringan.",
    "Komputer Paralel (Sistem Terdistribusi)": "Nilai akhir (0-100) Komputer Paralel & Sistem Komputasi Terdistribusi.",
    "Teknik Simulasi dan Pemodelan": "Nilai akhir (0-100) Teknik Simulasi & Pemodelan Sistem Nyata.",
    "Pengolahan Citra": "Nilai akhir (0-100) Pengolahan Citra Digital Lanjut & Segmentasi.",
    "Pemrograman Game": "Nilai akhir (0-100) Pemrograman Game & Game Engine Mechanics.",
    "Jaringan Syaraf Tiruan": "Nilai akhir (0-100) Jaringan Syaraf Tiruan (ANN / Deep Learning).",
    "Pemrograman Jaringan": "Nilai akhir (0-100) Pemrograman Jaringan Komputer & Socket API.",
    "Manajemen Proyek (Capstone Project)": "Nilai akhir (0-100) Manajemen Proyek (Capstone Project Informatika).",
    "Machine Learning": "Nilai akhir (0-100) Machine Learning & Evaluasi Algoritma Prediktif.",
    "Jaringan Komputer dan Komunikasi Data": "Nilai akhir (0-100) Jaringan Komputer & Komunikasi Data.",

    # IPS
    "IPS Semester 4": "Indeks Prestasi Semester 4 Anda (skala 0.00 - 4.00), dapat dilihat di KHS Semester 4.",
    "IPS Semester 5": "Indeks Prestasi Semester 5 Anda (skala 0.00 - 4.00), dapat dilihat di KHS Semester 5.",
    "IPS Semester 6": "Indeks Prestasi Semester 6 Anda (skala 0.00 - 4.00), dapat dilihat di KHS Semester 6.",
    "IPS Semester 7": "Indeks Prestasi Semester 7 Anda (skala 0.00 - 4.00), dapat dilihat di KHS Semester 7.",

    # 7 Minat Bidang (1 - 5)
    "Minat AI": "Ketertarikan pada Kecerdasan Buatan / Machine Learning / Deep Learning / Robotika Cerdas.",
    "Minat Data": "Ketertarikan pada Analisis Data, Data Mining, Business Intelligence, dan Eksplorasi Pola.",
    "Minat Software/Web": "Ketertarikan pada Perancangan Software, Pengembangan Web Fullstack, dan Rekayasa Aplikasi.",
    "Minat Keamanan Siber & Jaringan": "Ketertarikan pada Cybersecurity, Ethical Hacking, Enkripsi, dan Infrastruktur Jaringan Komputer.",
    "Minat Sistem Terdistribusi & Komputasi Paralel": "Ketertarikan pada Cloud Computing, Pemrosesan Paralel, Arsitektur Server Terdistribusi, dan Big Data Pipelines.",
    "Minat Teknologi Game & Multimedia": "Ketertarikan pada Game Development, Komputer Grafis, Augmented Reality (AR), dan Pengolahan Multimedia.",
    "Minat IoT & Sistem Embedded": "Ketertarikan pada Internet of Things (IoT), Mikrokontroler, Hardware Interfacing, dan Sistem Cerdas Tertanam.",

    # 6 Keterampilan Teknis (1 - 5)
    "Skill Python": "Tingkat kemahiran Python Anda (1 = pemula, 3 = program menengah, 5 = mahir library data/AI & project).",
    "Skill SQL": "Tingkat kemahiran SQL & Database Anda (1 = pemula, 3 = query dasar/JOIN, 5 = mahir/desain database).",
    "Skill Web Dev": "Tingkat kemahiran Web/Mobile Development (HTML/CSS/JS/Framework) (1 = pemula, 3 = bikin web interaktif, 5 = mahir fullstack).",
    "Skill OOP (Java/C++)": "Tingkat kemahiran Pemrograman Berorientasi Objek (OOP) Java/C++/C# (1 = pemula, 3 = inheritance/class dasar, 5 = design patterns & project kompleks).",
    "Skill Jaringan & Linux": "Tingkat kemahiran konfigurasi jaringan (IP, subnet, routing) & Linux CLI (1 = pemula, 3 = perintah terminal dasar, 5 = sysadmin/server setup).",
    "Skill Keamanan Siber": "Tingkat kemahiran tools keamanan siber (Wireshark, Nmap, dasar penetration testing) (1 = pemula, 3 = paham konsep vulnerabilitas, 5 = mahir security testing).",
}

WARNA = {"AI": "#6C5CE7", "Data Science": "#00B894", "RPL": "#E17055"}
DESKRIPSI = {
    "AI": "Jalur Peminatan Kecerdasan Buatan berfokus pada algoritma cerdas, computer vision, natural language processing, dan deep learning. Sangat cocok bagi mahasiswa dengan daya nalar matematis kuat dan antusiasme pada pemodelan AI.",
    "Data Science": "Jalur Peminatan Sains Data berfokus pada eksplorasi data, statistika inferensial, data mining, big data analytics, dan predictive modeling. Sangat cocok bagi mahasiswa yang teliti dan gemar mengungkap wawasan dari data.",
    "RPL": "Jalur Peminatan Rekayasa Perangkat Lunak berfokus pada arsitektur perangkat lunak modern, pengembangan web/mobile fullstack, pengujian sistem, dan metodologi software engineering modern.",
}

# ----------------------------------------------------------------------------
# FUNGSI UTILITAS FITUR
# ----------------------------------------------------------------------------
def get_fitur_semester(sem: int):
    """Menghasilkan daftar fitur lengkap untuk mahasiswa pada tingkatan semester tertentu (kumulatif)."""
    matkul = []
    ips = []
    for s in range(4, sem + 1):
        matkul.extend(MATKUL_PER_SEMESTER[s])
        ips.append(f"IPS Semester {s}")
    return matkul + ips + MINAT_COLS + SKILL_COLS

def get_semua_fitur():
    """Menghasilkan semua fitur kumulatif hingga semester 7."""
    return get_fitur_semester(7)

# ----------------------------------------------------------------------------
# DATA SINTETIS REALISTIS
# ----------------------------------------------------------------------------
def generate_data(n=800, seed=42):
    """Menghasilkan dataset sintetis profil mahasiswa informatika multi-semester."""
    rng = np.random.default_rng(seed)
    aff = rng.dirichlet([2, 2, 2], n)  # afinitas: AI, DS, RPL
    a_ai, a_ds, a_rpl = aff.T
    base = rng.normal(72, 6, n)

    def g(x):
        return np.clip(x + rng.normal(0, 5, n), 40, 100).round()

    def lk(a):
        return np.clip(np.round(1 + 4 * a * 1.6 + rng.normal(0, 0.6, n)), 1, 5)

    data = {
        # Semester 4 (2024/2025)
        "Komunikasi Interpersonal": g(base),
        "Matematika Informatika 3 (Logika Mat)": g(base + 22 * (a_ai - .33) + 12 * (a_ds - .33) - 10 * (a_rpl - .33)),
        "Pemrograman Berorientasi Objek": g(base + 24 * (a_rpl - .33) + 8 * (a_ds - .33)),
        "Pengantar Bahasa Dan Automata": g(base + 16 * (a_ai - .33) + 12 * (a_rpl - .33)),
        "Pengantar Interaksi Manusia Dan Komputer": g(base + 22 * (a_rpl - .33) - 8 * (a_ai - .33)),
        "Pengantar Metode Numerik": g(base + 20 * (a_ai - .33) + 20 * (a_ds - .33) - 15 * (a_rpl - .33)),
        "Pengantar Rangkaian Digital": g(base + 6 * (a_rpl - .33)),
        "Prak Pemrograman Berorientasi Objek": g(base + 25 * (a_rpl - .33) + 6 * (a_ds - .33)),
        "Rekayasa Perangkat Lunak": g(base + 28 * (a_rpl - .33) - 12 * (a_ai - .33)),

        # Semester 5 (2025/2026)
        "Etika Profesi Informatika": g(base),
        "Interface & Periferal": g(base + 10 * (a_rpl - .33)),
        "Keamanan Jaringan": g(base + 16 * (a_rpl - .33)),
        "Mikroprosessor & Mikrokontroler": g(base + 10 * (a_rpl - .33)),
        "Pemrograman Multimedia": g(base + 20 * (a_rpl - .33)),
        "Pengolahan Video": g(base + 18 * (a_ai - .33) + 10 * (a_rpl - .33)),
        "Prak Pemrograman Multimedia": g(base + 20 * (a_rpl - .33)),
        "RPL Berbasis Objek - Augmented Reality": g(base + 26 * (a_rpl - .33)),
        "Sistem Multimedia": g(base + 18 * (a_rpl - .33)),
        "Sistem Operasi": g(base + 15 * (a_rpl - .33) + 8 * (a_ai - .33)),
        "Teknik Kompilasi": g(base + 15 * (a_ai - .33) + 12 * (a_rpl - .33)),

        # Semester 6 (2025/2026)
        "Keamanan Cyber": g(base + 15 * (a_rpl - .33)),
        "Kecerdasan Buatan": g(base + 32 * (a_ai - .33) - 12 * (a_rpl - .33)),
        "Kerja Praktek / Penulisan Ilmiah": g(base),
        "Pemrograman Web": g(base + 30 * (a_rpl - .33) - 10 * (a_ai - .33)),
        "Pengantar Komputer Vision": g(base + 30 * (a_ai - .33) - 10 * (a_rpl - .33)),
        "Pengantar Sistem Informasi Manajemen": g(base + 18 * (a_ds - .33) + 12 * (a_rpl - .33)),
        "Prak Pemrograman Web": g(base + 30 * (a_rpl - .33) - 10 * (a_ai - .33)),
        "Probabilitas & Statistika": g(base + 28 * (a_ds - .33) + 20 * (a_ai - .33) - 18 * (a_rpl - .33)),
        "Riset Teknik Informasi / Metode Penelitian": g(base),
        "Teknik Data Mining": g(base + 32 * (a_ds - .33) - 12 * (a_rpl - .33)),
        "Wireless / Mobile Computing": g(base + 25 * (a_rpl - .33)),

        # Semester 7 (2026/2027)
        "Praktikum Data Mining": g(base + 32 * (a_ds - .33)),
        "Praktikum Jaringan": g(base + 16 * (a_rpl - .33)),
        "Komputer Paralel (Sistem Terdistribusi)": g(base + 18 * (a_ds - .33) + 15 * (a_rpl - .33)),
        "Teknik Simulasi dan Pemodelan": g(base + 22 * (a_ds - .33) + 18 * (a_ai - .33)),
        "Pengolahan Citra": g(base + 30 * (a_ai - .33)),
        "Pemrograman Game": g(base + 22 * (a_rpl - .33) + 12 * (a_ai - .33)),
        "Jaringan Syaraf Tiruan": g(base + 34 * (a_ai - .33) - 10 * (a_rpl - .33)),
        "Pemrograman Jaringan": g(base + 18 * (a_rpl - .33)),
        "Manajemen Proyek (Capstone Project)": g(base + 10 * (a_rpl - .33)),
        "Machine Learning": g(base + 30 * (a_ai - .33) + 22 * (a_ds - .33) - 15 * (a_rpl - .33)),
        "Jaringan Komputer dan Komunikasi Data": g(base + 16 * (a_rpl - .33)),
    }

    df = pd.DataFrame(data)

    # Menghitung IPS per semester realistis
    for s in [4, 5, 6, 7]:
        mats = MATKUL_PER_SEMESTER[s]
        avg_mat = df[mats].mean(axis=1)
        df[f"IPS Semester {s}"] = np.clip(2.5 + (avg_mat - 65) / 25 + rng.normal(0, 0.1, n), 2.0, 4.0).round(2)

    # 7 Domain Minat Bidang
    df["Minat AI"] = lk(a_ai)
    df["Minat Data"] = lk(a_ds)
    df["Minat Software/Web"] = lk(a_rpl)
    df["Minat Keamanan Siber & Jaringan"] = np.clip(np.round(1 + 4 * (0.65 * a_rpl + 0.25 * a_ds) * 1.6 + rng.normal(0, 0.5, n)), 1, 5)
    df["Minat Sistem Terdistribusi & Komputasi Paralel"] = np.clip(np.round(1 + 4 * (0.45 * a_ds + 0.45 * a_rpl + 0.2 * a_ai) * 1.5 + rng.normal(0, 0.5, n)), 1, 5)
    df["Minat Teknologi Game & Multimedia"] = np.clip(np.round(1 + 4 * (0.75 * a_rpl + 0.25 * a_ai) * 1.6 + rng.normal(0, 0.5, n)), 1, 5)
    df["Minat IoT & Sistem Embedded"] = np.clip(np.round(1 + 4 * (0.6 * a_rpl + 0.3 * a_ds) * 1.5 + rng.normal(0, 0.5, n)), 1, 5)

    # 6 Keterampilan Teknis Terapan
    df["Skill Python"] = np.clip(np.round(1 + 4 * (0.6 * a_ai + 0.4 * a_ds) * 1.6 + rng.normal(0, .5, n)), 1, 5)
    df["Skill SQL"] = lk(a_ds)
    df["Skill Web Dev"] = lk(a_rpl)
    df["Skill OOP (Java/C++)"] = np.clip(np.round(1 + 4 * (0.75 * a_rpl + 0.25 * a_ai) * 1.6 + rng.normal(0, .5, n)), 1, 5)
    df["Skill Jaringan & Linux"] = np.clip(np.round(1 + 4 * (0.65 * a_rpl + 0.3 * a_ds) * 1.5 + rng.normal(0, .5, n)), 1, 5)
    df["Skill Keamanan Siber"] = np.clip(np.round(1 + 4 * (0.7 * a_rpl + 0.2 * a_ds) * 1.5 + rng.normal(0, .5, n)), 1, 5)

    idx = aff.argmax(axis=1)
    flip = rng.random(n) < 0.10  # noise label 10%
    idx = np.where(flip, rng.integers(0, 3, n), idx)
    df["Peminatan"] = [KELAS[i] for i in idx]
    return df

# ----------------------------------------------------------------------------
# MODEL MULTI-SEMESTER TRAINING & CACHE
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan dataset dan melatih model klasifikasi multi-semester...")
def siapkan(versi="v4_7minat_6skill_robust"):
    """Melatih dan mengevaluasi model klasifikasi untuk setiap tingkatan semester (4 s/d 7)."""
    df = generate_data()
    eval_dict = {}
    model_dict = {}
    importansi_dict = {}
    Xte_dict = {}
    yte_dict = {}

    for s in [4, 5, 6, 7]:
        cols = get_fitur_semester(s)
        X, y = df[cols], df["Peminatan"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        kandidat = {
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        }
        hasil, models = [], {}
        for nama, m in kandidat.items():
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            p, r, f, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
            hasil.append({
                "Model": nama,
                "Accuracy": accuracy_score(yte, pred),
                "Precision": p,
                "Recall": r,
                "F1-Score": f,
            })
            models[nama] = (m, pred)

        eval_df = pd.DataFrame(hasil).sort_values("F1-Score", ascending=False).reset_index(drop=True)
        eval_dict[s] = eval_df
        model_dict[s] = models
        Xte_dict[s] = Xte
        yte_dict[s] = yte

        rf_model = models["Random Forest"][0]
        importansi_dict[s] = pd.Series(rf_model.feature_importances_, index=cols).sort_values(ascending=True)

    return df, eval_dict, model_dict, importansi_dict, Xte_dict, yte_dict

df, eval_dict, model_dict, importansi_dict, Xte_dict, yte_dict = siapkan()

def prediksi(nilai: dict, sem: int):
    """Melakukan inferensi peminatan dan probabilitas menggunakan model Random Forest semester terkait."""
    m_utama = model_dict[sem]["Random Forest"][0]
    cols = list(getattr(m_utama, "feature_names_in_", get_fitur_semester(sem)))
    row = {}
    for c in cols:
        if c in nilai:
            row[c] = nilai[c]
        elif "IPS" in c:
            row[c] = 3.25
        elif "Minat" in c or "Skill" in c:
            row[c] = 3
        else:
            row[c] = 75
    x = pd.DataFrame([row])[cols]
    proba = m_utama.predict_proba(x)[0]
    out = dict(zip(m_utama.classes_, proba))
    return max(out, key=out.get), out

def hitung_indeks_domain(nilai: dict, sem: int):
    """Menghitung agregat skor domain kompetensi, minat, dan skill untuk visualisasi radar chart."""
    mk_aktif = []
    for s in range(4, sem + 1):
        mk_aktif.extend(MATKUL_PER_SEMESTER[s])

    # Kategori mata kuliah per bidang
    ai_keys = [k for k in ["Kecerdasan Buatan", "Machine Learning", "Jaringan Syaraf Tiruan", "Pengolahan Citra",
                           "Pengantar Komputer Vision", "Matematika Informatika 3 (Logika Mat)", "Pengantar Metode Numerik",
                           "Pengantar Bahasa Dan Automata"] if k in mk_aktif]
    ds_keys = [k for k in ["Teknik Data Mining", "Praktikum Data Mining", "Probabilitas & Statistika",
                           "Pengantar Sistem Informasi Manajemen", "Teknik Simulasi dan Pemodelan",
                           "Komputer Paralel (Sistem Terdistribusi)"] if k in mk_aktif]
    rpl_keys = [k for k in ["Pemrograman Berorientasi Objek", "Prak Pemrograman Berorientasi Objek",
                            "Rekayasa Perangkat Lunak", "Pemrograman Web", "Prak Pemrograman Web",
                            "Pemrograman Multimedia", "RPL Berbasis Objek - Augmented Reality",
                            "Pemrograman Game", "Pengantar Interaksi Manusia Dan Komputer"] if k in mk_aktif]
    sis_keys = [k for k in ["Sistem Operasi", "Keamanan Jaringan", "Keamanan Cyber", "Pengantar Rangkaian Digital",
                            "Praktikum Jaringan", "Pemrograman Jaringan", "Jaringan Komputer dan Komunikasi Data",
                            "Interface & Periferal", "Mikroprosessor & Mikrokontroler"] if k in mk_aktif]

    ips_keys = [f"IPS Semester {s}" for s in range(4, sem + 1)]

    skor_ai = np.mean([nilai[k] for k in ai_keys]) if ai_keys else 70
    skor_ds = np.mean([nilai[k] for k in ds_keys]) if ds_keys else 70
    skor_rpl = np.mean([nilai[k] for k in rpl_keys]) if rpl_keys else 70
    skor_sis = np.mean([nilai[k] for k in sis_keys]) if sis_keys else 70
    rata_ips = np.mean([nilai[k] for k in ips_keys]) if ips_keys else 3.0

    return {
        "Akademik AI": skor_ai / 100.0,
        "Akademik Data": skor_ds / 100.0,
        "Akademik RPL": skor_rpl / 100.0,
        "Sistem & Jaringan": skor_sis / 100.0,
        "Prestasi IPS": rata_ips / 4.0,
        "Minat AI": nilai["Minat AI"] / 5.0,
        "Minat Data": nilai["Minat Data"] / 5.0,
        "Minat Software/Web": nilai["Minat Software/Web"] / 5.0,
        "Minat Cyber & Net": nilai["Minat Keamanan Siber & Jaringan"] / 5.0,
        "Minat Distr & Paralel": nilai["Minat Sistem Terdistribusi & Komputasi Paralel"] / 5.0,
        "Minat Game/Media": nilai["Minat Teknologi Game & Multimedia"] / 5.0,
        "Minat IoT/Embedded": nilai["Minat IoT & Sistem Embedded"] / 5.0,
        "Skill Python": nilai["Skill Python"] / 5.0,
        "Skill SQL": nilai["Skill SQL"] / 5.0,
        "Skill Web Dev": nilai["Skill Web Dev"] / 5.0,
        "Skill OOP": nilai["Skill OOP (Java/C++)"] / 5.0,
        "Skill Jaringan/Linux": nilai["Skill Jaringan & Linux"] / 5.0,
        "Skill Keamanan Siber": nilai["Skill Keamanan Siber"] / 5.0,
    }

def radar(nilai: dict, kelas: str, sem: int):
    """Membuat visualisasi radar chart indeks kompetensi mahasiswa vs rata-rata peminatan."""
    user_indeks = hitung_indeks_domain(nilai, sem)
    cols = list(user_indeks.keys())
    user_vals = list(user_indeks.values())

    df_kelas = df[df.Peminatan == kelas]
    avg_nilai = {c: df_kelas[c].mean() for c in get_fitur_semester(sem)}
    kelas_indeks = hitung_indeks_domain(avg_nilai, sem)
    kelas_vals = [kelas_indeks[c] for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_vals + user_vals[:1],
        theta=cols + cols[:1],
        fill="toself",
        name="Profil Anda",
        line_color="#0984e3",
    ))
    fig.add_trace(go.Scatterpolar(
        r=kelas_vals + kelas_vals[:1],
        theta=cols + cols[:1],
        fill="toself",
        name=f"Rata-rata Mahasiswa {kelas}",
        line_color=WARNA[kelas],
        opacity=0.45,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=480,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig

def bar_proba(probs: dict, judul="Distribusi Probabilitas Peminatan"):
    """Membuat bar chart probabilitas peminatan horizontal."""
    d = pd.DataFrame({"Peminatan": list(probs), "Probabilitas": [v * 100 for v in probs.values()]})
    fig = px.bar(
        d, x="Probabilitas", y="Peminatan", orientation="h", color="Peminatan",
        color_discrete_map=WARNA, text=d["Probabilitas"].round(1).astype(str) + "%", title=judul
    )
    fig.update_layout(showlegend=False, xaxis_range=[0, 100], height=260, margin=dict(t=45, b=10))
    return fig

def tampil_panduan():
    """Menampilkan panduan langkah pengisian data akademik semester 4-7 dan skala penilaian."""
    st.markdown("### 📋 Panduan Langkah Pengisian")
    st.markdown(
        "1. **Pilih Semester Saat Ini (4, 5, 6, atau 7):**\n"
        "   - **Semester 4:** Mahasiswa mengisi nilai mata kuliah Semester 4 dan IPS Semester 4.\n"
        "   - **Semester 5:** Mahasiswa mengisi nilai mata kuliah Semester 4 & 5, serta IPS Semester 4 & 5.\n"
        "   - **Semester 6:** Mahasiswa mengisi nilai mata kuliah Semester 4, 5 & 6, serta IPS Semester 4, 5 & 6.\n"
        "   - **Semester 7:** Mahasiswa mengisi nilai mata kuliah Semester 4, 5, 6 & 7, serta IPS Semester 4, 5, 6 & 7.\n"
        "2. **Siapkan KHS / Transkrip Nilai:** Masukkan nilai angka (0-100). Jika pada transkrip berupa huruf:\n"
        "   - `A`: 85 - 100 | `B`: 70 - 84 | `C`: 55 - 69 | `D`: 40 - 54\n"
        "3. **Isi IPS (Indeks Prestasi Semester):** Masukkan nilai IPS pada skala 0.00 - 4.00 untuk tiap semester yang telah diselesaikan. Sistem otomatis menghitung estimasi IPK kumulatif.\n"
        "4. **Isi 7 Minat Bidang (1-5):** Nilai ketertarikan Anda pada domain AI, Data, Software/Web, Cybersecurity & Jaringan, Sistem Terdistribusi, Game & Multimedia, serta IoT & Embedded.\n"
        "5. **Isi 6 Keterampilan Teknis (1-5):** Ukur kemahiran hands-on pada Python, SQL, Web Dev, OOP (Java/C++), Jaringan/Linux, dan Keamanan Siber.\n"
        "6. **Klik 'Dapatkan Rekomendasi'** untuk melihat hasil analisis, grafik radar kompetensi, dan mencoba mode simulasi What-If."
    )

    st.markdown("### 📊 Panduan Interpretasi Skala 1 - 5")
    st.dataframe(pd.DataFrame({
        "Skala": [1, 2, 3, 4, 5],
        "Tingkat Minat": ["Tidak tertarik sama sekali", "Kurang tertarik", "Cukup tertarik", "Tertarik", "Sangat antusias / tertarik"],
        "Tingkat Keterampilan": [
            "Pemula / belum pernah mencoba",
            "Paham konsep dasar",
            "Mampu membuat program / query sederhana",
            "Cukup mahir / terbiasa menyelesaikan problem",
            "Mahir / pernah membuat aplikasi / project nyata",
        ],
    }), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# NAVIGASI SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("🎓 SIMPEMI")
st.sidebar.caption("Sistem Rekomendasi Peminatan & Matkul Pilihan")
st.sidebar.markdown("**Capstone Design Informatika**")
st.sidebar.divider()

halaman = st.sidebar.radio(
    "Menu Navigasi",
    [
        "🏠 Beranda",
        "📖 Panduan Pengisian",
        "🎯 Simulasi Rekomendasi",
        "📊 Evaluasi Model",
        "🗂️ Dataset",
        "ℹ️ Tentang",
    ],
)
st.sidebar.divider()
st.sidebar.info("💡 Sistem ini adalah **pendukung keputusan cerdas**. Hasil rekomendasi dapat didiskusikan lebih lanjut dengan Dosen Wali / Pembimbing Akademik.")

# ----------------------------------------------------------------------------
# 1. BERANDA
# ----------------------------------------------------------------------------
if halaman == "🏠 Beranda":
    st.title("🎓 SIMPEMI - Rekomendasi Peminatan & Matkul Pilihan")
    st.subheader("Sistem Simulasi Berbasis Machine Learning untuk Mahasiswa Informatika (Semester 4 - 7)")

    st.markdown(
        "Aplikasi **SIMPEMI** dirancang untuk membantu mahasiswa Program Studi Informatika dalam menentukan "
        "jalur peminatan terbaik (**Kecerdasan Buatan / AI**, **Sains Data / Data Science**, atau **Rekayasa Perangkat Lunak / RPL**) "
        "serta rekomendasi mata kuliah pilihan tingkat lanjut yang selaras dengan rekam jejak akademik, minat, dan keahlian teknis."
    )

    df_riwayat_count = len(muat_riwayat_mahasiswa())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mahasiswa Terdata (Riil)", f"{df_riwayat_count} Orang")
    c2.metric("Tingkat Semester", "Semester 4 s/d 7")
    c3.metric("Minat & Skill", f"{len(MINAT_COLS)} Minat + {len(SKILL_COLS)} Skill")
    rf_acc_sem7 = eval_dict[7].loc[eval_dict[7].Model == "Random Forest", "Accuracy"].iloc[0]
    c4.metric("Akurasi Model Utama", f"{rf_acc_sem7:.1%}")

    st.divider()
    st.markdown("### 🌟 Fitur Unggulan Sistem")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("#### 🔄 Pengisian Kumulatif Dinamis")
        st.write("Mendukung mahasiswa dari Semester 4 hingga Semester 7. Formulir secara otomatis menyesuaikan daftar mata kuliah dan IPS sesuai progres semester yang dipilih.")
    with k2:
        st.markdown("#### 📈 Berbasis IPS per Semester")
        st.write("Menggantikan isian IPK statis dengan input IPS (Indeks Prestasi Semester) spesifik tiap semester serta kalkulasi estimasi IPK kumulatif secara otomatis.")
    with k3:
        st.markdown("#### 🛠️ 6 Keterampilan Teknis Terapan")
        st.write("Mengukur kemampuan praktis Python, SQL, Web Dev, OOP (Java/C++), Jaringan/Linux CLI, dan Keamanan Siber.")

    st.divider()
    st.markdown("### 🎯 Profil Jalur Peminatan")
    col_k = st.columns(3)
    for col, k in zip(col_k, KELAS):
        with col:
            st.markdown(f"#### {k}")
            st.write(DESKRIPSI[k])
            st.markdown("**Contoh Matkul Pilihan Lanjutan:**")
            for mk in MATKUL[k][:3]:
                st.caption(f"• {mk}")

# ----------------------------------------------------------------------------
# 2. PANDUAN PENGISIAN
# ----------------------------------------------------------------------------
elif halaman == "📖 Panduan Pengisian":
    st.title("📖 Panduan Pengisian Data Akademik")
    st.write("Silakan pelajari petunjuk pengisian berikut agar rekomendasi yang dihasilkan akurat dan representatif.")
    tampil_panduan()

    st.divider()
    st.markdown("### Penjelasan 7 Domain Minat Bidang Informatika")
    tabel_minat = pd.DataFrame([{"Domain Minat": mk, "Penjelasan & Ruang Lingkup": HELP.get(mk, "-")} for mk in MINAT_COLS])
    st.dataframe(tabel_minat, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🛠️ Penjelasan 6 Keterampilan Teknis Terapan")
    tabel_skill = pd.DataFrame([{"Keterampilan Teknis": sk, "Deskripsi & Ekspektasi Praktis": HELP.get(sk, "-")} for sk in SKILL_COLS])
    st.dataframe(tabel_skill, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📚 Daftar Mata Kuliah per Semester (Kurikulum Informatika)")
    t_sem4, t_sem5, t_sem6, t_sem7 = st.tabs(["Semester 4", "Semester 5", "Semester 6", "Semester 7"])

    with t_sem4:
        st.markdown(f"**Tahun Akademik: {TA_SEMESTER[4]} (9 Mata Kuliah + IPS Semester 4)**")
        tabel4 = pd.DataFrame([{"Mata Kuliah / Isian": mk, "Deskripsi": HELP.get(mk, "-")} for mk in MATKUL_PER_SEMESTER[4] + ["IPS Semester 4"]])
        st.dataframe(tabel4, use_container_width=True, hide_index=True)

    with t_sem5:
        st.markdown(f"**Tahun Akademik: {TA_SEMESTER[5]} (11 Mata Kuliah + IPS Semester 5)**")
        tabel5 = pd.DataFrame([{"Mata Kuliah / Isian": mk, "Deskripsi": HELP.get(mk, "-")} for mk in MATKUL_PER_SEMESTER[5] + ["IPS Semester 5"]])
        st.dataframe(tabel5, use_container_width=True, hide_index=True)

    with t_sem6:
        st.markdown(f"**Tahun Akademik: {TA_SEMESTER[6]} (11 Mata Kuliah + IPS Semester 6)**")
        tabel6 = pd.DataFrame([{"Mata Kuliah / Isian": mk, "Deskripsi": HELP.get(mk, "-")} for mk in MATKUL_PER_SEMESTER[6] + ["IPS Semester 6"]])
        st.dataframe(tabel6, use_container_width=True, hide_index=True)

    with t_sem7:
        st.markdown(f"**Tahun Akademik: {TA_SEMESTER[7]} (11 Mata Kuliah + IPS Semester 7)**")
        tabel7 = pd.DataFrame([{"Mata Kuliah / Isian": mk, "Deskripsi": HELP.get(mk, "-")} for mk in MATKUL_PER_SEMESTER[7] + ["IPS Semester 7"]])
        st.dataframe(tabel7, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# 3. SIMULASI REKOMENDASI
# ----------------------------------------------------------------------------
elif halaman == "🎯 Simulasi Rekomendasi":
    st.title("🎯 Simulasi Rekomendasi Peminatan")
    st.write("Pilih semester Anda saat ini, isi nilai akademik secara kumulatif beserta IPS tiap semester, tingkat minat, dan keterampilan teknis.")

    with st.expander("📖 Bingung cara mengisi? Klik di sini untuk melihat ringkasan panduan", expanded=False):
        tampil_panduan()

    # 1. Pilih Semester
    pilih_sem = st.radio(
        "📌 **Pilih Semester Anda Saat Ini:**",
        [4, 5, 6, 7],
        horizontal=True,
        format_func=lambda s: f"Semester {s} ({TA_SEMESTER[s]})",
        help="Sistem akan menampilkan formulir pengisian mata kuliah dan IPS secara kumulatif dari Semester 4 hingga semester yang Anda pilih.",
    )

    # 2. Preset Cepat
    contoh = st.selectbox(
        "⚡ Opsi Pengisian Cepat dengan Profil Preset (Opsional):",
        ["Isi manual", "Profil condong AI", "Profil condong Data Science", "Profil condong RPL"],
    )

    def generate_preset_values(tipe: str, sem_target: int):
        val = {}
        for s in range(4, sem_target + 1):
            for mk in MATKUL_PER_SEMESTER[s]:
                if tipe == "Profil condong AI":
                    if mk in ["Kecerdasan Buatan", "Machine Learning", "Jaringan Syaraf Tiruan", "Pengolahan Citra",
                              "Pengantar Komputer Vision", "Matematika Informatika 3 (Logika Mat)", "Pengantar Metode Numerik"]:
                        val[mk] = 92
                    elif mk in ["Teknik Data Mining", "Praktikum Data Mining", "Probabilitas & Statistika"]:
                        val[mk] = 80
                    elif mk in ["Rekayasa Perangkat Lunak", "Pemrograman Web", "Prak Pemrograman Web", "Pemrograman Game"]:
                        val[mk] = 68
                    else:
                        val[mk] = 78
                elif tipe == "Profil condong Data Science":
                    if mk in ["Teknik Data Mining", "Praktikum Data Mining", "Probabilitas & Statistika",
                              "Pengantar Sistem Informasi Manajemen", "Teknik Simulasi dan Pemodelan"]:
                        val[mk] = 93
                    elif mk in ["Kecerdasan Buatan", "Machine Learning", "Matematika Informatika 3 (Logika Mat)"]:
                        val[mk] = 82
                    elif mk in ["Rekayasa Perangkat Lunak", "Pemrograman Web", "Prak Pemrograman Web", "Pemrograman Multimedia"]:
                        val[mk] = 68
                    else:
                        val[mk] = 78
                elif tipe == "Profil condong RPL":
                    if mk in ["Pemrograman Berorientasi Objek", "Prak Pemrograman Berorientasi Objek", "Rekayasa Perangkat Lunak",
                              "Pemrograman Web", "Prak Pemrograman Web", "RPL Berbasis Objek - Augmented Reality",
                              "Pemrograman Multimedia", "Pemrograman Game", "Sistem Operasi"]:
                        val[mk] = 92
                    elif mk in ["Teknik Data Mining", "Probabilitas & Statistika"]:
                        val[mk] = 72
                    elif mk in ["Kecerdasan Buatan", "Jaringan Syaraf Tiruan", "Pengantar Metode Numerik"]:
                        val[mk] = 65
                    else:
                        val[mk] = 78
                else:
                    val[mk] = 75

            # IPS preset
            if tipe == "Profil condong AI":
                val[f"IPS Semester {s}"] = 3.75
            elif tipe == "Profil condong Data Science":
                val[f"IPS Semester {s}"] = 3.70
            elif tipe == "Profil condong RPL":
                val[f"IPS Semester {s}"] = 3.70
            else:
                val[f"IPS Semester {s}"] = 3.25

        # 7 Minat Preset
        if tipe == "Profil condong AI":
            val["Minat AI"] = 5
            val["Minat Data"] = 3
            val["Minat Software/Web"] = 2
            val["Minat Keamanan Siber & Jaringan"] = 2
            val["Minat Sistem Terdistribusi & Komputasi Paralel"] = 4
            val["Minat Teknologi Game & Multimedia"] = 3
            val["Minat IoT & Sistem Embedded"] = 2

            val["Skill Python"] = 5
            val["Skill SQL"] = 3
            val["Skill Web Dev"] = 2
            val["Skill OOP (Java/C++)"] = 4
            val["Skill Jaringan & Linux"] = 3
            val["Skill Keamanan Siber"] = 2
        elif tipe == "Profil condong Data Science":
            val["Minat AI"] = 3
            val["Minat Data"] = 5
            val["Minat Software/Web"] = 2
            val["Minat Keamanan Siber & Jaringan"] = 2
            val["Minat Sistem Terdistribusi & Komputasi Paralel"] = 5
            val["Minat Teknologi Game & Multimedia"] = 2
            val["Minat IoT & Sistem Embedded"] = 3

            val["Skill Python"] = 4
            val["Skill SQL"] = 5
            val["Skill Web Dev"] = 2
            val["Skill OOP (Java/C++)"] = 3
            val["Skill Jaringan & Linux"] = 3
            val["Skill Keamanan Siber"] = 2
        elif tipe == "Profil condong RPL":
            val["Minat AI"] = 2
            val["Minat Data"] = 2
            val["Minat Software/Web"] = 5
            val["Minat Keamanan Siber & Jaringan"] = 5
            val["Minat Sistem Terdistribusi & Komputasi Paralel"] = 4
            val["Minat Teknologi Game & Multimedia"] = 5
            val["Minat IoT & Sistem Embedded"] = 4

            val["Skill Python"] = 3
            val["Skill SQL"] = 4
            val["Skill Web Dev"] = 5
            val["Skill OOP (Java/C++)"] = 5
            val["Skill Jaringan & Linux"] = 4
            val["Skill Keamanan Siber"] = 4
        else:
            for k in MINAT_COLS:
                val[k] = 3
            for k in SKILL_COLS:
                val[k] = 3

        return val

    preset_saat_ini = generate_preset_values(contoh, pilih_sem)

    # FORM PENGISIAN
    with st.form("form_rekomendasi"):
        nilai_input = {}

        # IDENTITAS MAHASISWA
        st.markdown("### 👤 Identitas Mahasiswa")
        st.caption("Data identitas wajib diisi untuk mencatat riwayat simulasi ke dalam sistem database:")
        c_id1, c_id2, c_id3 = st.columns(3)
        with c_id1:
            nama_input = st.text_input("Nama Lengkap Mahasiswa *", value="", placeholder="Contoh: Muhammad Adam", help="Nama lengkap mahasiswa.")
        with c_id2:
            nim_input = st.text_input("NIM (Nomor Induk Mahasiswa) *", value="", placeholder="Contoh: 50421001", help="Nomor Induk Mahasiswa aktif.")
        with c_id3:
            kelas_input = st.text_input("Kelas / Rombel *", value="", placeholder="Contoh: 4IA01 / 5IA02", help="Kelas perkuliahan mahasiswa.")

        st.divider()

        st.markdown(f"### 📚 Data Akademik (Kumulatif Semester 4 s/d {pilih_sem})")
        st.caption("Masukkan nilai angka (0-100) untuk mata kuliah dan IPS (0.00 - 4.00) untuk masing-masing semester.")

        for s in range(4, pilih_sem + 1):
            st.markdown(f"#### 📅 Semester {s} (Tahun Akademik {TA_SEMESTER[s]})")

            # Input IPS Semester
            ips_key = f"IPS Semester {s}"
            ips_col1, _ = st.columns([1, 2])
            with ips_col1:
                nilai_input[ips_key] = st.number_input(
                    f"🏆 {ips_key}",
                    min_value=0.0,
                    max_value=4.0,
                    value=float(preset_saat_ini.get(ips_key, 3.25)),
                    step=0.01,
                    format="%.2f",
                    help=HELP[ips_key],
                )

            # Sliders Mata Kuliah Semester Ini (3 Kolom)
            mk_list = MATKUL_PER_SEMESTER[s]
            cols = st.columns(3)
            for idx, mk in enumerate(mk_list):
                with cols[idx % 3]:
                    nilai_input[mk] = st.slider(
                        mk,
                        min_value=0,
                        max_value=100,
                        value=int(preset_saat_ini.get(mk, 75)),
                        help=HELP.get(mk, f"Nilai akhir mata kuliah {mk}."),
                        key=f"input_{s}_{idx}",
                    )
            st.divider()

        # MINAT BIDANG (7 DOMAIN)
        st.markdown("### Minat Bidang (Skala 1 = Rendah, 5 = Tinggi)")
        st.caption("Tentukan tingkat ketertarikan Anda pada masing-masing domain peminatan informatika:")

        # Baris 1: 3 Bidang Utama
        c_minat1 = st.columns(3)
        with c_minat1[0]:
            nilai_input["Minat AI"] = st.slider(
                "Minat AI (Kecerdasan Buatan)",
                1, 5, int(preset_saat_ini.get("Minat AI", 3)),
                help=HELP["Minat AI"], key="minat_ai",
            )
        with c_minat1[1]:
            nilai_input["Minat Data"] = st.slider(
                "Minat Data (Sains Data)",
                1, 5, int(preset_saat_ini.get("Minat Data", 3)),
                help=HELP["Minat Data"], key="minat_data",
            )
        with c_minat1[2]:
            nilai_input["Minat Software/Web"] = st.slider(
                "Minat Software & Web (RPL)",
                1, 5, int(preset_saat_ini.get("Minat Software/Web", 3)),
                help=HELP["Minat Software/Web"], key="minat_software",
            )

        # Baris 2: 4 Bidang Lanjutan
        c_minat2 = st.columns(4)
        with c_minat2[0]:
            nilai_input["Minat Keamanan Siber & Jaringan"] = st.slider(
                "Keamanan Siber & Jaringan",
                1, 5, int(preset_saat_ini.get("Minat Keamanan Siber & Jaringan", 3)),
                help=HELP["Minat Keamanan Siber & Jaringan"], key="minat_cyber",
            )
        with c_minat2[1]:
            nilai_input["Minat Sistem Terdistribusi & Komputasi Paralel"] = st.slider(
                "Sistem Terdistribusi & Paralel",
                1, 5, int(preset_saat_ini.get("Minat Sistem Terdistribusi & Komputasi Paralel", 3)),
                help=HELP["Minat Sistem Terdistribusi & Komputasi Paralel"], key="minat_distr",
            )
        with c_minat2[2]:
            nilai_input["Minat Teknologi Game & Multimedia"] = st.slider(
                "Teknologi Game & Multimedia",
                1, 5, int(preset_saat_ini.get("Minat Teknologi Game & Multimedia", 3)),
                help=HELP["Minat Teknologi Game & Multimedia"], key="minat_game",
            )
        with c_minat2[3]:
            nilai_input["Minat IoT & Sistem Embedded"] = st.slider(
                "IoT & Sistem Embedded",
                1, 5, int(preset_saat_ini.get("Minat IoT & Sistem Embedded", 3)),
                help=HELP["Minat IoT & Sistem Embedded"], key="minat_iot",
            )

        st.divider()

        # KETERAMPILAN TEKNIS (6 KETERAMPILAN TERAPAN)
        st.markdown("### 🛠️ Keterampilan Teknis (Skala 1 = Pemula, 5 = Mahir)")
        st.caption("Ukur kemampuan praktis hands-on coding, database, dan infrastruktur komputasi Anda:")

        # Baris 1: 3 Keterampilan Inti
        c_sk1 = st.columns(3)
        with c_sk1[0]:
            nilai_input["Skill Python"] = st.slider(
                "Skill Python (AI & Data)",
                1, 5, int(preset_saat_ini.get("Skill Python", 3)),
                help=HELP["Skill Python"], key="sk_python",
            )
        with c_sk1[1]:
            nilai_input["Skill SQL"] = st.slider(
                "Skill SQL & Basis Data",
                1, 5, int(preset_saat_ini.get("Skill SQL", 3)),
                help=HELP["Skill SQL"], key="sk_sql",
            )
        with c_sk1[2]:
            nilai_input["Skill Web Dev"] = st.slider(
                "Skill Web & Mobile Dev",
                1, 5, int(preset_saat_ini.get("Skill Web Dev", 3)),
                help=HELP["Skill Web Dev"], key="sk_web",
            )

        # Baris 2: 3 Keterampilan Engineering & Sistem
        c_sk2 = st.columns(3)
        with c_sk2[0]:
            nilai_input["Skill OOP (Java/C++)"] = st.slider(
                "Skill OOP (Java/C++)",
                1, 5, int(preset_saat_ini.get("Skill OOP (Java/C++)", 3)),
                help=HELP["Skill OOP (Java/C++)"], key="sk_oop",
            )
        with c_sk2[1]:
            nilai_input["Skill Jaringan & Linux"] = st.slider(
                "Skill Jaringan & Linux CLI",
                1, 5, int(preset_saat_ini.get("Skill Jaringan & Linux", 3)),
                help=HELP["Skill Jaringan & Linux"], key="sk_net",
            )
        with c_sk2[2]:
            nilai_input["Skill Keamanan Siber"] = st.slider(
                "Skill Keamanan Siber",
                1, 5, int(preset_saat_ini.get("Skill Keamanan Siber", 3)),
                help=HELP["Skill Keamanan Siber"], key="sk_cyber",
            )

        tombol_kirim = st.form_submit_button("🔍 Dapatkan Rekomendasi Peminatan", type="primary", use_container_width=True)

    if tombol_kirim:
        if not nama_input.strip() or not nim_input.strip() or not kelas_input.strip():
            st.error("⚠️ **Wajib Mengisi Identitas:** Mohon lengkapi Nama Lengkap, NIM, dan Kelas terlebih dahulu sebelum mendapatkan rekomendasi!")
        else:
            waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ips_aktif = [nilai_input[f"IPS Semester {s}"] for s in range(4, pilih_sem + 1)]
            ipk_est = np.mean(ips_aktif)
            terbaik, probs = prediksi(nilai_input, pilih_sem)

            # Simpan secara otomatis ke file riwayat mahasiswa (CSV)
            record_entry = {
                "Waktu": waktu_sekarang,
                "Nama": nama_input.strip(),
                "NIM": nim_input.strip(),
                "Kelas": kelas_input.strip(),
                "Semester": pilih_sem,
                "Estimasi_IPK": round(ipk_est, 2),
                "Rekomendasi_Peminatan": terbaik,
                "Keyakinan": f"{probs[terbaik]:.1%}",
                "Prob_AI": round(probs["AI"], 4),
                "Prob_Data_Science": round(probs["Data Science"], 4),
                "Prob_RPL": round(probs["RPL"], 4),
            }
            for k_col, v_val in nilai_input.items():
                record_entry[k_col] = v_val

            simpan_riwayat_mahasiswa(record_entry)

            st.session_state["data_rekomendasi"] = {
                "nama": nama_input.strip(),
                "nim": nim_input.strip(),
                "kelas": kelas_input.strip(),
                "nilai": nilai_input,
                "semester": pilih_sem,
                "waktu": waktu_sekarang,
                "terbaik": terbaik,
                "probs": probs,
                "ipk_est": ipk_est,
            }

    # HASIL REKOMENDASI
    if "data_rekomendasi" in st.session_state:
        stored_data = st.session_state["data_rekomendasi"]
        nama_mhs = stored_data.get("nama", "Mahasiswa")
        nim_mhs = stored_data.get("nim", "-")
        kelas_mhs = stored_data.get("kelas", "-")
        waktu_sim = stored_data.get("waktu", "-")
        nilai = stored_data["nilai"]
        sem_digunakan = stored_data["semester"]
        ipk_est = stored_data.get("ipk_est", np.mean([nilai[f"IPS Semester {s}"] for s in range(4, sem_digunakan + 1)]))
        terbaik = stored_data.get("terbaik")
        probs = stored_data.get("probs")

        if terbaik is None or probs is None:
            terbaik, probs = prediksi(nilai, sem_digunakan)

        urut = sorted(probs, key=probs.get, reverse=True)

        st.divider()
        st.success(f"💾 **Data Telah Tersimpan ke Dataset Riwayat Mahasiswa:** **{nama_mhs}** | NIM: **{nim_mhs}** | Kelas: **{kelas_mhs}** | Waktu: *{waktu_sim}*")
        st.markdown(f"## ✅ Rekomendasi Peminatan Terbaik: :{WARNA[terbaik].lower()}[{terbaik}] ({probs[terbaik]:.1%})")
        st.markdown(f"> **Profil Akademik:** Mahasiswa Semester {sem_digunakan} | Estimasi IPK Kumulatif: **{ipk_est:.2f}**")
        st.write(DESKRIPSI[terbaik])

        if probs[urut[0]] - probs[urut[1]] < 0.10:
            st.warning(
                f"⚠️ **Peringatan Selisih Tipis:** Selisih probabilitas antara **{urut[0]}** ({probs[urut[0]]:.1%}) "
                f"dan **{urut[1]}** ({probs[urut[1]]:.1%}) kurang dari 10%. "
                "Kedua jalur peminatan ini sangat relevan untuk Anda. Diskusikan lebih lanjut dengan Dosen Wali."
            )

        col_kiri, col_kanan = st.columns([1, 1])
        with col_kiri:
            st.plotly_chart(bar_proba(probs), use_container_width=True)
            st.markdown(f"#### 📖 Rekomendasi Mata Kuliah Pilihan ({terbaik})")
            for m in MATKUL[terbaik]:
                st.markdown(f"- **{m}**")
            st.caption(f"Alternatif Peminatan Kedua ({urut[1]}): " + ", ".join(MATKUL[urut[1]][:3]))

        with col_kanan:
            st.markdown("#### 🕸️ Radar Indeks Kompetensi Profil Mahasiswa")
            st.plotly_chart(radar(nilai, terbaik, sem_digunakan), use_container_width=True)

        # Feature Importance untuk Semester Tersebut
        st.markdown(f"#### 🔎 Faktor Fitur Paling Berpengaruh pada Model Semester {sem_digunakan}")
        top_imp = importansi_dict[sem_digunakan].tail(10)
        fig_imp = px.bar(
            x=top_imp.values,
            y=top_imp.index,
            orientation="h",
            labels={"x": "Tingkat Pengaruh (Importance)", "y": "Fitur"},
            color=top_imp.values,
            color_continuous_scale="Purples",
        )
        fig_imp.update_layout(height=360, margin=dict(t=15, b=10), showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        # ---------------- MODE SIMULASI WHAT-IF ----------------
        st.divider()
        st.markdown("## 🧪 Mode Simulasi Skenario What-If")
        st.caption(
            "Uji coba skenario: Bagaimana jika Anda meningkatkan nilai pada mata kuliah kunci atau mengasah keterampilan teknis? "
            "Sistem akan langsung memproyeksikan pergeseran peluang rekomendasi."
        )

        mk_whatif = []
        if sem_digunakan >= 4:
            mk_whatif.append("Matematika Informatika 3 (Logika Mat)")
            mk_whatif.append("Rekayasa Perangkat Lunak")
        if sem_digunakan >= 5:
            mk_whatif.append("Sistem Operasi")
        if sem_digunakan >= 6:
            mk_whatif.append("Kecerdasan Buatan")
            mk_whatif.append("Teknik Data Mining")
            mk_whatif.append("Pemrograman Web")
        if sem_digunakan >= 7:
            mk_whatif.append("Machine Learning")
            mk_whatif.append("Praktikum Data Mining")

        mk_whatif_selected = mk_whatif[-4:] if len(mk_whatif) > 4 else mk_whatif

        st.markdown("**1. Simulasi Nilai Mata Kuliah Kunci:**")
        col_w = st.columns(len(mk_whatif_selected))
        sim_nilai = dict(nilai)

        for i, mk in enumerate(mk_whatif_selected):
            sim_nilai[mk] = col_w[i].slider(
                f"{mk}",
                0,
                100,
                int(nilai[mk]),
                key=f"whatif_mk_{mk}",
            )

        st.markdown("**2. Simulasi Keterampilan Teknis Terapan:**")
        col_w2 = st.columns(3)
        for i, k in enumerate(SKILL_COLS[:3]):
            sim_nilai[k] = col_w2[i].slider(
                f"{k}",
                1,
                5,
                int(nilai[k]),
                key=f"whatif_sk_{k}",
            )

        col_w3 = st.columns(3)
        for i, k in enumerate(SKILL_COLS[3:]):
            sim_nilai[k] = col_w3[i].slider(
                f"{k}",
                1,
                5,
                int(nilai[k]),
                key=f"whatif_sk_{k}",
            )

        t2, p2 = prediksi(sim_nilai, sem_digunakan)

        d1, d2 = st.columns(2)
        d1.metric("Rekomendasi Profil Asli", terbaik, f"{probs[terbaik]:.1%}")
        d2.metric(
            "Rekomendasi Hasil Simulasi",
            t2,
            f"{(p2[t2] - probs[t2]):+.1%} vs profil asli" if t2 == terbaik else f"{p2[t2]:.1%}",
        )

        df_cmp = pd.DataFrame({
            "Peminatan": KELAS * 2,
            "Skenario": ["Asli"] * 3 + ["Simulasi"] * 3,
            "Probabilitas": [probs[k] * 100 for k in KELAS] + [p2[k] * 100 for k in KELAS],
        })
        fig_cmp = px.bar(
            df_cmp,
            x="Peminatan",
            y="Probabilitas",
            color="Skenario",
            barmode="group",
            color_discrete_sequence=["#b2bec3", "#0984e3"],
            title="Perbandingan Probabilitas: Skenario Asli vs Simulasi",
        )
        fig_cmp.update_layout(height=320, yaxis_range=[0, 100])
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Unduh Hasil CSV
        ringkas_data = [{
            "Semester": sem_digunakan,
            "Estimasi_IPK": round(ipk_est, 2),
            **nilai,
            "Rekomendasi": terbaik,
            **{f"P({k})": round(probs[k], 4) for k in KELAS},
        }]
        df_export = pd.DataFrame(ringkas_data)
        st.download_button(
            "⬇️ Unduh Hasil Rekomendasi (CSV)",
            df_export.to_csv(index=False).encode("utf-8"),
            "hasil_rekomendasi_simpemi.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.info("💡 Silakan sesuaikan data akademik, nilai IPS, minat bidang, dan keterampilan teknis pada formulir di atas, lalu klik tombol **🔍 Dapatkan Rekomendasi Peminatan**.")

# ----------------------------------------------------------------------------
# 4. EVALUASI MODEL
# ----------------------------------------------------------------------------
elif halaman == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model Klasifikasi")
    st.caption("Membandingkan performa 4 algoritma machine learning (Random Forest, Logistic Regression, KNN, Decision Tree) untuk tiap semester.")

    pilih_eval_sem = st.selectbox(
        "Pilih Tingkatan Model Semester:",
        [4, 5, 6, 7],
        format_func=lambda s: f"Model Semester {s} ({len(get_fitur_semester(s))} Fitur)",
    )

    tabel_eval_sem = eval_dict[pilih_eval_sem]
    model_terlatih_sem = model_dict[pilih_eval_sem]
    yte_sem = yte_dict[pilih_eval_sem]
    imp_sem = importansi_dict[pilih_eval_sem]

    st.markdown(f"#### 🏆 Tabel Metrik Evaluasi Model Semester {pilih_eval_sem}")
    st.dataframe(
        tabel_eval_sem.style.format({c: "{:.2%}" for c in tabel_eval_sem.columns[1:]}),
        use_container_width=True,
        hide_index=True,
    )

    fig_eval = px.bar(
        tabel_eval_sem.melt(id_vars="Model", var_name="Metrik", value_name="Nilai"),
        x="Model",
        y="Nilai",
        color="Metrik",
        barmode="group",
        title=f"Perbandingan Metrik Evaluasi (Model Semester {pilih_eval_sem})",
    )
    fig_eval.update_layout(yaxis_range=[0, 1], height=380)
    st.plotly_chart(fig_eval, use_container_width=True)

    c_cm, c_fi = st.columns([1, 1])
    with c_cm:
        pilih_alg = st.selectbox("Confusion Matrix untuk Model:", list(model_terlatih_sem.keys()), index=0)
        cm = confusion_matrix(yte_sem, model_terlatih_sem[pilih_alg][1], labels=KELAS)
        fig_cm = px.imshow(
            cm,
            x=KELAS,
            y=KELAS,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah Sampel"),
            title=f"Confusion Matrix ({pilih_alg} - Sem {pilih_eval_sem})",
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_fi:
        st.markdown(f"**Top 10 Feature Importance (Random Forest Semester {pilih_eval_sem})**")
        top_10_imp = imp_sem.tail(10)
        fig_fi = px.bar(
            x=top_10_imp.values,
            y=top_10_imp.index,
            orientation="h",
            labels={"x": "Importance", "y": ""},
            color=top_10_imp.values,
            color_continuous_scale="Purples",
        )
        fig_fi.update_layout(height=400, margin=dict(t=30), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ----------------------------------------------------------------------------
# 5. DATASET
# ----------------------------------------------------------------------------
elif halaman == "🗂️ Dataset":
    st.title("🗂️ Dataset & Riwayat Mahasiswa")
    st.write("Halaman ini menyediakan data riwayat pengisian simulasi oleh mahasiswa secara riil serta dataset acuan pelatihan model AI.")

    tab_riwayat, tab_latih = st.tabs(["📋 Data Riwayat Mahasiswa (Hasil Simulasi)", "🧠 Dataset Latih AI (800 Sampel Acuan)"])

    with tab_riwayat:
        st.markdown("### 📋 Data Mahasiswa yang Telah Mengisi Simulasi")
        st.caption("Setiap mahasiswa yang mengisi form simulasi akan tercatat secara permanen di sini.")

        df_riwayat = muat_riwayat_mahasiswa()

        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        c_r1.metric("Total Mahasiswa Mengisi", f"{len(df_riwayat)} Mahasiswa")
        if len(df_riwayat) > 0 and "Rekomendasi_Peminatan" in df_riwayat.columns:
            c_r2.metric("Rekomendasi AI", len(df_riwayat[df_riwayat.Rekomendasi_Peminatan == "AI"]))
            c_r3.metric("Rekomendasi Data Science", len(df_riwayat[df_riwayat.Rekomendasi_Peminatan == "Data Science"]))
            c_r4.metric("Rekomendasi RPL", len(df_riwayat[df_riwayat.Rekomendasi_Peminatan == "RPL"]))
        else:
            c_r2.metric("Rekomendasi AI", "0")
            c_r3.metric("Rekomendasi Data Science", "0")
            c_r4.metric("Rekomendasi RPL", "0")

        if len(df_riwayat) > 0:
            st.divider()
            col_search, _ = st.columns([2, 1])
            with col_search:
                kata_kunci = st.text_input("🔍 Cari Mahasiswa (Nama / NIM / Kelas / Peminatan):", placeholder="Ketik kata kunci pencarian...")

            df_tampil = df_riwayat
            if kata_kunci.strip():
                kw = kata_kunci.strip().lower()
                mask = (
                    df_tampil["Nama"].astype(str).str.lower().str.contains(kw) |
                    df_tampil["NIM"].astype(str).str.lower().str.contains(kw) |
                    df_tampil["Kelas"].astype(str).str.lower().str.contains(kw) |
                    df_tampil["Rekomendasi_Peminatan"].astype(str).str.lower().str.contains(kw)
                )
                df_tampil = df_tampil[mask]

            kolom_ringkas = [c for c in ["Waktu", "Nama", "NIM", "Kelas", "Semester", "Estimasi_IPK", "Rekomendasi_Peminatan", "Keyakinan"] if c in df_tampil.columns]
            st.dataframe(df_tampil[kolom_ringkas], use_container_width=True, hide_index=True)

            with st.expander("🔍 Lihat Detail Lengkap Semua Kolom (Nilai, IPS, Minat, Skill)", expanded=False):
                st.dataframe(df_tampil, use_container_width=True, hide_index=True)

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                st.download_button(
                    "⬇️ Unduh Data Riwayat Mahasiswa (CSV)",
                    df_riwayat.to_csv(index=False).encode("utf-8"),
                    "data_riwayat_mahasiswa_simpemi.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with c_btn2:
                with st.popover("🗑️ Hapus / Reset Seluruh Riwayat"):
                    st.warning("Apakah Anda yakin ingin menghapus seluruh data riwayat mahasiswa yang tersimpan?")
                    if st.button("Ya, Hapus Semua Data Riwayat", type="primary"):
                        hapus_riwayat_mahasiswa()
                        st.success("Riwayat data mahasiswa berhasil dihapus.")
                        st.rerun()
        else:
            st.info("ℹ️ Belum ada data mahasiswa yang tersimpan. Silakan buka menu **🎯 Simulasi Rekomendasi**, isi identitas (Nama, NIM, Kelas) dan nilai Anda, lalu klik **Dapatkan Rekomendasi Peminatan**.")

    with tab_latih:
        st.markdown("### 🧠 Dataset Latih AI (800 Sampel Acuan)")
        st.info(
            "Dataset terdiri dari 800 data mahasiswa sintetis dengan aturan afinitas akademik, 7 domain minat, dan 6 keterampilan teknis, "
            "dilengkapi variasi noise acak ±10% untuk mensimulasikan kondisi riil perkuliahan Informatika."
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"#### 📄 Pratinjau Data ({len(df)} baris x {df.shape[1]} kolom)")
            kolom_filter = st.selectbox(
                "Tampilkan kolom:",
                ["Semua Kolom", "Semester 4", "Semester 5", "Semester 6", "Semester 7", "IPS per Semester", "Minat & Keterampilan"],
            )
            if kolom_filter == "Semester 4":
                tampil_df = df[MATKUL_PER_SEMESTER[4] + ["IPS Semester 4", "Peminatan"]]
            elif kolom_filter == "Semester 5":
                tampil_df = df[MATKUL_PER_SEMESTER[5] + ["IPS Semester 5", "Peminatan"]]
            elif kolom_filter == "Semester 6":
                tampil_df = df[MATKUL_PER_SEMESTER[6] + ["IPS Semester 6", "Peminatan"]]
            elif kolom_filter == "Semester 7":
                tampil_df = df[MATKUL_PER_SEMESTER[7] + ["IPS Semester 7", "Peminatan"]]
            elif kolom_filter == "IPS per Semester":
                tampil_df = df[[f"IPS Semester {s}" for s in range(4, 8)] + ["Peminatan"]]
            elif kolom_filter == "Minat & Keterampilan":
                tampil_df = df[MINAT_COLS + SKILL_COLS + ["Peminatan"]]
            else:
                tampil_df = df

            st.dataframe(tampil_df.head(100), use_container_width=True)

        with c2:
            st.markdown("#### 🎯 Distribusi Label Peminatan")
            fig_pie = px.pie(
                df, names="Peminatan", color="Peminatan", color_discrete_map=WARNA,
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📈 Statistik Deskriptif Fitur")
        st.dataframe(df[get_semua_fitur()].describe().T.round(2), use_container_width=True)

        fitur_pilih = st.selectbox("Visualisasikan distribusi fitur per peminatan:", get_semua_fitur())
        fig_hist = px.histogram(
            df, x=fitur_pilih, color="Peminatan", barmode="overlay", opacity=0.65,
            color_discrete_map=WARNA, title=f"Distribusi Fitur: {fitur_pilih}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.download_button(
            "⬇️ Unduh Dataset Latih Lengkap (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "dataset_simpemi_multi_semester.csv",
            "text/csv",
        )

# ----------------------------------------------------------------------------
# 6. TENTANG
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ Tentang SIMPEMI")
    st.markdown("""
### Sistem Informasi Rekomendasi Peminatan & Mata Kuliah Pilihan (SIMPEMI)
Aplikasi berbasis Web Decision Support System (DSS) yang memanfaatkan algoritma Machine Learning untuk merekomendasikan
jalur peminatan (**Kecerdasan Buatan (AI)**, **Sains Data**, atau **Rekayasa Perangkat Lunak (RPL)**) bagi mahasiswa Program Studi Informatika.

#### 📌 Pembaruan & Fitur Utama:
1. **Dukungan Mahasiswa Semester 4 hingga 7:**
   - Formulir dinamis dan kumulatif yang menyesuaikan progres semester mahasiswa.
   - Mahasiswa Semester 4 mengisi mata kuliah Semester 4 dan IPS Semester 4.
   - Mahasiswa Semester 5 mengisi Semester 4 & 5 serta IPS Semester 4 & 5.
   - Mahasiswa Semester 6 mengisi Semester 4, 5 & 6 serta IPS Semester 4, 5 & 6.
   - Mahasiswa Semester 7 mengisi Semester 4, 5, 6 & 7 serta IPS Semester 4, 5, 6 & 7.
2. **Kalkulasi IPS & Estimasi IPK:**
   - Input IPK statis digantikan dengan input **IPS (Indeks Prestasi Semester)** individual untuk tiap semester.
   - Sistem secara otomatis menghitung estimasi IPK kumulatif.
3. **7 Domain Minat Bidang Informatika Terintegrasi:**
   - Kecerdasan Buatan (AI)
   - Sains Data (Data Science)
   - Software & Web (Rekayasa Perangkat Lunak)
   - Keamanan Siber & Jaringan Komputer (Cybersecurity & Networking)
   - Sistem Terdistribusi & Komputasi Paralel (Distributed Systems & Parallel Computing)
   - Teknologi Game & Multimedia (Game Development & Computer Graphics)
   - Internet of Things (IoT) & Sistem Embedded
4. **6 Keterampilan Teknis Terapan:**
   - Skill Python (AI & Data Science)
   - Skill SQL & Basis Data
   - Skill Web & Mobile Dev
   - Skill Pemrograman OOP (Java/C++)
   - Skill Jaringan Komputer & Linux CLI
   - Skill Keamanan Siber (Security Tools)
5. **Model Klasifikasi Multi-Tier:**
   - Melatih model machine learning independen untuk setiap jenjang semester (Semester 4, 5, 6, 7), memastikan akurasi inferensi optimal tanpa kebocoran data atau missing value.
6. **Simulasi Skenario What-If & Radar Chart Kompetensi:**
   - Visualisasi perbandingan profil kompetensi mahasiswa dengan rata-rata peminatan.
   - Fasilitas simulasi what-if untuk mengeksplorasi strategi perbaikan nilai dan keahlian teknis.

---
- **Mata Kuliah:** Manajemen Proyek / Capstone Project Informatika
- **Teknologi:** Python, Streamlit, Scikit-learn, Pandas, NumPy, Plotly
- **Model Utama:** Random Forest Classifier (dibandingkan dengan Logistic Regression, KNN, dan Decision Tree)
- **Keterbatasan:** Dataset menggunakan data sintetis dengan distribusi realistis. Rekomendasi bersifat sebagai sistem pendukung keputusan (decision support), bukan penentu mutlak.
""")
