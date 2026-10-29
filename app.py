# Bagian ini digunakan untuk mengimpor library atau pustaka Python yang dibutuhkan aplikasi
import re # menghapus URL, angka, tanda baca, dan spasi berlebih.
import joblib # memuat model machine learning yang sudah dilatih dan disimpan dalam file .pkl.
import requests # mengirim permintaan HTTP ke alamat URL berita
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup # membaca struktur HTML halaman web dan mengekstrak teks berita dari elemen HTML tertentu
from pathlib import Path


# ============================================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="centered"
)

# Custom Styling Tampilan UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .card-sport {
        background-color: #EFF6FF;
        border-left: 6px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .card-finance {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
#  1. LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model_w2v = joblib.load( # memuat model Word2Vec yg sudah dibuat
    MODEL_DIR / "w2v_model.pkl"
)

model_nb = joblib.load( # memuat model naive bayes
    MODEL_DIR / "nb_model.pkl"
)


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocessing(teks):

    teks = str(teks).lower() # mengubah input menjadi string dan mengubah huruf menjadi kecil

    # Menghapus URL
    teks = re.sub(r"http\S+|www\S+", "", teks)

    # Menghapus angka
    teks = re.sub(r"\d+", "", teks)

    # Menghapus tanda baca
    teks = re.sub(r"[^a-z\s]", " ", teks)

    # Menghapus spasi berlebih
    teks = re.sub(r"\s+", " ", teks).strip()

    return teks


# ============================================================
# 3. MENGUBAH TEKS MENJADI VEKTOR MENGGUNAKAN WORD2VEC SKIP GRAM
# ============================================================

def document_vector(tokens, model): # fungsi menerima dua parameter dari tokens dan model yg sudah dilatih

    vectors = [] # daftar kosong untuk menampung vektor tiap kata

    for word in tokens: # memeriksa kata dalam token

        if word in model.wv: # memeriksa kata ada dalam model word3vec skip gram
            vectors.append(model.wv[word]) # mengumpulkan vektor kata tsb

    if len(vectors) == 0: # jika tidak ada kata yg dikenal model, maka vektor 0
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# ============================================================
# 4. MENGAMBIL ISI BERITA DARI URL
# ============================================================

def ambil_berita_dari_url(url): # fungsi mengambil isi berita dari url

    headers = {
        "User-Agent": "Mozilla/5.0" # informasi identitas klien, membantu mengakases situs
    }

    response = requests.get( # mengirim HTTP GET request ke URL.
        url,
        headers=headers,
        timeout=15 # menetapkan batas waktu tunggu 15 detik.
    )

    response.raise_for_status() # menyimpan hasil permintaan

    soup = BeautifulSoup(
        response.text, # berisi HTML halaman web yg diterima
        "html.parser"
    )

    # Selector utama Detik & Kompas
    isi = soup.select_one(
        "div.detail__body-text"
    )

    # Selector Kompas
    if isi is None:
        isi = soup.select_one(
            "div.read__content"
        )

    # Selector alternatif, lokasi artikel sesuai struktur html situs yg diakses
    if isi is None:
        isi = soup.select_one(
            "div.detail__body"
        )

    if isi is None:
        isi = soup.select_one(
            "div.itp_bodycontent"
        )

    if isi is None:
        isi = soup.select_one(
            "article"
        )

    if isi is None:
        paragraphs = soup.find_all("p")
        teks_paragraf = " ".join([p.get_text(strip=True) for p in paragraphs])
        return teks_paragraf if len(teks_paragraf) > 100 else None

    return isi.get_text( # mengambil teks di dalam html
        separator=" ", # memberikan spasi antar paragraf
        strip=True # menghapus spasi berlebih diawal dan akhir teks
    )


# ============================================================
# 5. KLASIFIKASI BERITA MENGGUNAKAN NAIVE BAYES
# ============================================================

def klasifikasi_berita(teks): # fungsi mengklasifikasi berita ke dalam kategori sport dan finance

    teks_bersih = preprocessing(teks) # fungsi preprocessing untuk pemberihan teks

    tokens = teks_bersih.split() # membagi teks menjadi perkata/token

    vektor = document_vector( # mengubah token menjadi vektor dgn model word2vec skip-gram
        tokens, # kumpulan kata dari berita yg sudah dibersihkan
        model_w2v # model word2vec skip gram yg sudah dimuat
    )

    # Mengambil probabilitas setiap kategori
    probabilitas = model_nb.predict_proba( # memasukkan vektor berita ke model naive bayes untuk menghitung probabilitas setiap kategor sport dan finance
        [vektor]
    )[0]

    # Mengambil kategori dengan probabilitas tertinggi
    indeks = np.argmax(probabilitas) # mencaari posisi probabilitas tertinggi

    hasil = model_nb.classes_[indeks] # mengambil nama kategori berdasarkan posisi tsb

    confidence = probabilitas[indeks] # mengambil probabilitas dari kategori yg dipilih

    return hasil, confidence # fungsi untuk mengembalikan kategori harsil prediksi dan probabilitas


# ============================================================
# 6. VALIDASI KATEGORI BERDASARKAN URL
# ============================================================

def cek_kategori_url(url):

    url = url.lower() # teks diubah menjadi huruf kecil

    if any(kategori in url for kategori in ["sport", "bola", "olahraga"]): # mengecek apakah url mengandung kata kunci kategori
        return "sport"

    elif any(kategori in url for kategori in ["finance", "ekonomi", "money", "bisnis"]): # mengecek apakah url mengandung kata kunci
        return "finance"

    else:
        return None


# ============================================================
# 7. TAMPILAN APLIKASI
# ============================================================

st.markdown('<p class="main-title">📰 Klasifikasi Berita</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasi klasifikasi berita menggunakan <b>Skip-gram Word2Vec dan Naive Bayes</b>.</p>', unsafe_allow_html=True)


# ============================================================
# 8. INPUT URL
# ============================================================

url = st.text_input( # memasukkan url berita 
    "🔗 URL Berita",
    placeholder="https://sport.detik.com/... atau https://finance.detik.com/..." # contoh url berita yg muncul di dalm kolom input
)


# ============================================================
# 9. TOMBOL KLASIFIKASI
# ============================================================

if st.button(
    "🔍 Klasifikasi Berita",
    use_container_width=True,
    type="primary"
):

    if url.strip() == "": # menampilkan peringatan jika klom input kosong

        st.warning(
            "Silakan masukkan URL berita terlebih dahulu."
        )

    else:

        # Mengecek kategori URL
        kategori_url = cek_kategori_url(url) # menyimpan url yg dimasukin di kolom input ke fungsi cek_kategori_url 6

        if kategori_url is None: # jika kategori url tidak sesuai dengan kategori

            st.warning( 
                "⚠️ Kategori berita tidak didukung."
            )

            st.write(
                "Aplikasi hanya dapat mengklasifikasikan "
                "berita dengan kategori SPORT dan FINANCE."
            )

        else:

            try:

                with st.spinner( # menampilkan animasi loading saat proses berlangsung
                    "Mengambil dan memproses berita..." # pesan yg muncul
                ):

                    berita = ambil_berita_dari_url( # jika url lolos maka akan masuk ke fungsi ini 4
                        url # link berita yg dimasukkan di kolom input
                    )

                if berita is None: # jika berita kosong

                    st.error( # maka akan menampilkan pesan error
                        "Isi berita tidak ditemukan."
                    )

                else: # jika isi berita ditemukan

                    hasil, confidence = klasifikasi_berita( # memanggil fungsi klasifikasi berita untuk mengklasifikasikan isi berita 5
                        berita # variabel yang berisi teks artikel hasil pengambilan dari URL yang dimasukkan pengguna
                    )

                    st.success( # menampilkan pesan sukses jika berita berhasil diklasifikasikan
                        "Berita berhasil diklasifikasikan!"
                    )

                    st.subheader( # menampilkan subjudul hasil klasifikasi
                        "Hasil Klasifikasi"
                    )

                    # Tampilan Kartu Hasil Klasifikasi
                    if hasil.lower() == "sport":

                        st.markdown(
                            f"""
                            <div class="card-sport">
                                <h3 style="margin:0; color:#1E40AF;">🏆 Kategori: SPORT</h3>
                                <p style="margin:5px 0 0 0; color:#2563EB; font-weight:600;">Tingkat Akurasi: {confidence * 100:.2f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    elif hasil.lower() == "finance":

                        st.markdown(
                            f"""
                            <div class="card-finance">
                                <h3 style="margin:0; color:#065F46;">💰 Kategori: FINANCE</h3>
                                <p style="margin:5px 0 0 0; color:#10B981; font-weight:600;">Tingkat Akurasi: {confidence * 100:.2f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with st.expander(
                        "📄 Lihat isi berita"
                    ):

                        st.write(berita)

            except requests.exceptions.RequestException:

                st.error(
                    "URL tidak dapat diakses. "
                    "Pastikan URL berita Detik benar."
                )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan: {e}"
                )