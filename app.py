# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Prediksi Permintaan Fashion",
    page_icon="👕",
    layout="wide"
)

# === FUNGSI-FUNGSI BANTUAN ===

@st.cache_data
def load_data(url):
    """Memuat dan membersihkan data dari CSV."""
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    # Hanya fokus pada kategori Fashion/Clothing
    df_clothing = df[df['Category'] == 'Clothing'].copy()
    df_clothing.dropna(inplace=True)
    return df_clothing

@st.cache_data
def prep_for_model(df):
    """Mempersiapkan data untuk pelatihan model."""
    # Membuat target variable (klasifikasi)
    median_sold = df['Units Sold'].median()
    df['Demand'] = df['Units Sold'].apply(lambda x: 'Tinggi' if x > median_sold else 'Rendah')
    
    # Memilih fitur untuk model
    features = ['Inventory Level', 'Weather Condition', 'Holiday/Promotion']
    target = 'Demand'
    
    X = df[features]
    y = df[target]
    
    return X, y, median_sold

@st.cache_resource
def train_and_save_models(X, y):
    """Melatih, mengevaluasi, dan menyimpan model KNN dan Naive Bayes."""
    # Memisahkan fitur numerik dan kategorikal
    numeric_features = ['Inventory Level']
    categorical_features = ['Weather Condition', 'Holiday/Promotion']

    # Membuat preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Pipeline untuk KNN
    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', KNeighborsClassifier(n_neighbors=5))])

    # Pipeline untuk Naive Bayes
    nb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', GaussianNB())])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Latih model
    knn_pipeline.fit(X_train, y_train)
    nb_pipeline.fit(X_train, y_train)
    
    # Simpan model
    joblib.dump(knn_pipeline, 'knn_model.joblib')
    joblib.dump(nb_pipeline, 'nb_model.joblib')

    # Evaluasi
    y_pred_knn = knn_pipeline.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    
    y_pred_nb = nb_pipeline.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
    
    return acc_knn, report_knn, acc_nb, report_nb

@st.cache_resource
def load_model(path):
    """Memuat model yang sudah dilatih dari file."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

# === DEFINISI HALAMAN-HALAMAN APLIKASI ===

def show_homepage():
    st.title("👕 Dashboard Prediksi Permintaan Produk Fashion")
    st.markdown(
        """
        Selamat datang di Dashboard Prediksi Permintaan Produk Fashion.
        
        Gunakan menu di sebelah kiri untuk bernavigasi antar halaman:
        
        ### 1. 📊 EDA dan Karakteristik
        Halaman ini menampilkan analisis data eksplorasi (EDA) dari dataset inventaris ritel untuk kategori **pakaian**. 
        Anda dapat melihat dataset, statistik deskriptif, dan visualisasi untuk memahami pola data.
        
        ### 2. 🤖 Hasil Pelatihan Model
        Halaman ini menunjukkan hasil dari proses pelatihan model. Klik tombol untuk melatih model
        KNN dan Naive Bayes yang akan digunakan untuk prediksi.
        
        ### 3. 🔮 Formulir Prediksi
        Gunakan formulir interaktif di halaman ini untuk memasukkan data baru dan mendapatkan prediksi permintaan 
        secara real-time.
        ---
        """
    )

def show_eda(df):
    st.title("📊 Halaman 1: Exploratory Data Analysis (EDA)")
    st.markdown("Halaman ini menampilkan analisis, statistik, dan visualisasi dari dataset.")

    st.header("Tampilan Awal Dataset (Kategori: Pakaian)")
    st.dataframe(df.head())

    st.header("Karakteristik dan Statistik Data")
    st.write(df.describe())

    st.header("Distribusi Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Jumlah Baris Data", value=f"{df.shape[0]:,}")
    with col2:
        st.metric(label="Jumlah Kolom Data", value=df.shape[1])
        
    st.header("Visualisasi Data")
    
    st.subheader("Tren Penjualan Harian (Units Sold)")
    time_series_fig = px.line(df, x='Date', y='Units Sold', title='Penjualan Harian Seiring Waktu', 
                              labels={'Date': 'Tanggal', 'Units Sold': 'Unit Terjual'})
    st.plotly_chart(time_series_fig, use_container_width=True)

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("Penjualan Berdasarkan Kondisi Cuaca")
        weather_fig = px.box(df, x='Weather Condition', y='Units Sold', title='Distribusi Penjualan vs Cuaca',
                              labels={'Weather Condition': 'Kondisi Cuaca', 'Units Sold': 'Unit Terjual'})
        st.plotly_chart(weather_fig, use_container_width=True)
        
    with col_vis2:
        st.subheader("Penjualan Selama Promosi/Hari Libur")
        promo_fig = px.pie(df, names='Holiday/Promotion', title='Proporsi Penjualan pada Hari Promosi/Libur', 
                           hole=0.3, labels={'Holiday/Promotion': 'Promosi/Libur'})
        st.plotly_chart(promo_fig, use_container_width=True)

def show_training_results(df):
    st.title("🤖 Halaman 2: Hasil Pelatihan Model")
    st.markdown("Halaman ini menjelaskan proses dan menampilkan hasil dari pelatihan model.")

    X, y, median_val = prep_for_model(df.copy())

    st.header("1. Persiapan Data")
    st.markdown(f"""
    - **Fitur (Input)**: `Inventory Level`, `Weather Condition`, `Holiday/Promotion`.
    - **Target (Output)**: `Demand` (Permintaan).
    - Variabel target `Demand` dibuat secara biner:
        - **Tinggi**: Jika `Units Sold` > {int(median_val)}
        - **Rendah**: Jika `Units Sold` <= {int(median_val)}
    """)

    st.header("2. Pelatihan dan Evaluasi Model")
    if st.button("Latih Model Sekarang", type="primary"):
        with st.spinner("Mohon tunggu, model sedang dilatih..."):
            acc_knn, report_knn, acc_nb, report_nb = train_and_save_models(X, y)
            st.session_state['models_trained'] = True
            st.session_state['results'] = (acc_knn, report_knn, acc_nb, report_nb)
        st.success("Model berhasil dilatih dan disimpan!")

    if 'models_trained' in st.session_state:
        acc_knn, report_knn, acc_nb, report_nb = st.session_state['results']
        st.subheader("Hasil Evaluasi")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("K-Nearest Neighbors (KNN)")
            st.metric("Akurasi", f"{acc_knn:.2%}")
            st.text("Laporan Klasifikasi:")
            st.dataframe(pd.DataFrame(report_knn).transpose())
        
        with col2:
            st.subheader("Naive Bayes")
            st.metric("Akurasi", f"{acc_nb:.2%}")
            st.text("Laporan Klasifikasi:")
            st.dataframe(pd.DataFrame(report_nb).transpose())

def show_prediction_form():
    st.title("🔮 Halaman 3: Formulir Prediksi Permintaan")
    st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi permintaan produk fashion.")

    knn_model = load_model('knn_model.joblib')
    nb_model = load_model('nb_model.joblib')

    if not knn_model or not nb_model:
        st.warning("Model belum dilatih. Silakan pergi ke halaman **'Hasil Pelatihan Model'** dan klik tombol 'Latih Model Sekarang' terlebih dahulu.")
        return

    weather_options = ['Clear', 'Cloudy', 'Rainy', 'Stormy']
    promo_options = [0, 1]

    with st.form("prediction_form"):
        st.header("Masukkan Data untuk Prediksi")
        col1, col2 = st.columns(2)
        
        with col1:
            inventory = st.number_input("Tingkat Inventaris (Stok Awal)", min_value=0, max_value=1000, value=100, step=10)
            weather = st.selectbox("Kondisi Cuaca", options=weather_options, index=0)
        
        with col2:
            promotion = st.selectbox("Ada Hari Libur/Promosi?", options=promo_options, format_func=lambda x: "Ya" if x == 1 else "Tidak", index=1)
        
        submit_button = st.form_submit_button(label="🚀 Lakukan Prediksi")

    if submit_button:
        input_data = pd.DataFrame({
            'Inventory Level': [inventory],
            'Weather Condition': [weather],
            'Holiday/Promotion': [promotion]
        })
        
        st.subheader("Hasil Prediksi")
        
        pred_knn = knn_model.predict(input_data)[0]
        prob_knn = knn_model.predict_proba(input_data)
        
        pred_nb = nb_model.predict(input_data)[0]
        prob_nb = nb_model.predict_proba(input_data)
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.info("Prediksi dari Model KNN")
            if pred_knn == "Tinggi":
                st.success(f"📈 Permintaan Diprediksi **{pred_knn}**")
            else:
                st.warning(f"📉 Permintaan Diprediksi **{pred_knn}**")
            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prob_knn, columns=knn_model.classes_, index=['Prob.']))

        with col_res2:
            st.info("Prediksi dari Model Naive Bayes")
            if pred_nb == "Tinggi":
                st.success(f"📈 Permintaan Diprediksi **{pred_nb}**")
            else:
                st.warning(f"📉 Permintaan Diprediksi **{pred_nb}**")
            st.write("Probabilitas:")
            st.dataframe(pd.DataFrame(prob_nb, columns=nb_model.classes_, index=['Prob.']))


# === NAVIGASI UTAMA APLIKASI ===

def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ("Beranda", "📊 EDA dan Karakteristik", "🤖 Hasil Pelatihan Model", "🔮 Formulir Prediksi")
    )
    
    try:
        df = load_data('retail_inventory.csv')

        if page == "Beranda":
            show_homepage()
        elif page == "📊 EDA dan Karakteristik":
            show_eda(df)
        elif page == "🤖 Hasil Pelatihan Model":
            show_training_results(df)
        elif page == "🔮 Formulir Prediksi":
            show_prediction_form()

    except FileNotFoundError:
        st.error("File `retail_inventory.csv` tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan `app.py`.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")


if __name__ == "__main__":
    main()

