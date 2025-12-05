import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="KLASIFIKASI KATEGORI BERITA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling yang lebih menarik
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .main-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .card-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .card-content {
        color: #555;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #764ba2;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    .step-number {
        background: #667eea;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -------------------------------
# CLEANER
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

# -------------------------------
# SIDEBAR MENU
# -------------------------------
st.sidebar.markdown("### 🎯 PILIH MENU")
menu = st.sidebar.selectbox(
    "",
    ["🏠 Home", "📂 Upload Dataset", "🧠 Train Model", "🔮 Predict News"],
    label_visibility="collapsed"
)
# -------------------------------
# HOME
# -------------------------------
if menu == "🏠 Home":
    # Header utama
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">📰 BBC News Classification</h1>
            <p class="main-subtitle">Klasifikasi Kategori Berita menggunakan Convolutional Neural Networks (CNN)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Deskripsi aplikasi
    st.markdown("""
        <div class="card">
            <div class="card-title">📋 Tentang Aplikasi</div>
            <div class="card-content">
                Aplikasi ini adalah sistem klasifikasi teks otomatis yang menggunakan 
                <strong>Deep Learning</strong> dengan arsitektur <strong>Convolutional Neural Networks (CNN)</strong> 
                untuk mengkategorikan berita BBC ke dalam berbagai kategori seperti Sport, Business, Politics, 
                Tech, dan Entertainment.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Fitur utama
    st.markdown("""
        <div class="card">
            <div class="card-title">✨ Fitur Utama</div>
            <div class="card-content">
                <div class="feature-box">
                    <strong>📤 Upload & Preprocessing Dataset</strong><br>
                    Upload file CSV dan lakukan pembersihan teks secara otomatis
                </div>
                <div class="feature-box">
                    <strong>🎓 Training Model CNN</strong><br>
                    Latih model dengan parameter yang dapat disesuaikan
                </div>
                <div class="feature-box">
                    <strong>📊 Visualisasi Performa</strong><br>
                    Lihat akurasi, confusion matrix, dan classification report
                </div>
                <div class="feature-box">
                    <strong>🔍 Prediksi Berita Baru</strong><br>
                    Klasifikasikan kategori berita dari teks yang Anda masukkan
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Panduan penggunaan
    col1, = st.columns(1)
    
    with col1:
        st.markdown("""
            <div class="card">
                <div class="card-title">ℹ️ Petunjuk Penggunaaan</div>
                <div class="card-content">
                    <ul>
                        <li>Masukan dataset yang memiliki kolom category sebagai label dan kolom text berita</li>
                        <li>Training dataset dengan epoch minimal 5 agar hasil yang didapatkan maksimal</li>
                        <li>Masukan teks berita berbahasa Inggris untuk diklasifikasikan</li>
                    </ul>
                    <p><strong>Format:</strong> CSV (category, text)</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            <strong>💡 Tips:</strong> Untuk hasil terbaik, gunakan minimal 5 epochs saat training. 
            Semakin banyak epochs, semakin baik akurasi model (tetapi waktu training lebih lama).
        </div>
    """, unsafe_allow_html=True)
    
    # Warning box
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Catatan:</strong> Pastikan file dataset memiliki kolom "category" dan "text". 
            Training model memerlukan waktu beberapa menit tergantung spesifikasi komputer Anda.
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# UPLOAD DATASET
# -------------------------------
elif menu == "📂 Upload Dataset":
    st.title("📂 Upload Dataset BBC")

    file = st.file_uploader("Upload file bbc-text.csv", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.rename(columns={"category": "labels"}, inplace=True)
        df["clean_text"] = df["text"].apply(clean_text)

        st.success("✅ Dataset berhasil dimuat!")
        
        # Tampilkan info dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Jumlah Kategori", df["labels"].nunique())
        with col3:
            st.metric("Kolom", len(df.columns))
        
        st.subheader("📊 Preview Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Distribusi Kategori")
        fig, ax = plt.subplots(figsize=(8, 4))
        df["labels"].value_counts().plot(kind="bar", ax=ax, color="#667eea")
        ax.set_title("Jumlah Data per Kategori")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

        st.session_state["df"] = df

# -------------------------------
# TRAIN MODEL
# -------------------------------
elif menu == "🧠 Train Model":
    st.title("🧠 Train CNN Model")

    if "df" not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu.")
    else:
        df = st.session_state["df"]

        # Encode labels
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["labels"])
        num_classes = df["label"].nunique()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"],
            df["label"],
            test_size=0.2,
            stratify=df["label"],
            random_state=42,
        )

        MAX_WORDS = 20000
        MAX_LEN = 300

        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        st.write("### ⚙️ Parameter Training")
        epochs = st.slider("Jumlah Epochs", 2, 20, 5)
        
        st.info(f"📊 Data Training: {len(X_train)} | Data Testing: {len(X_test)}")

        if st.button("🚀 Mulai Training", type="primary"):
            with st.spinner("🔄 Training model... Mohon tunggu..."):
                model = Sequential([
                    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
                    Conv1D(128, 5, activation="relu"),
                    GlobalMaxPooling1D(),
                    Dropout(0.5),
                    Dense(64, activation="relu"),
                    Dense(num_classes, activation="softmax")
                ])

                model.compile(
                    loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"]
                )

                history = model.fit(
                    X_train_pad, y_train_cat,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0
                )

            # Save model & tokenizer
            st.session_state["model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.session_state["label_encoder"] = le

            st.success("✅ Training selesai!")

            # Show accuracy plot
            st.subheader("📈 Grafik Akurasi Training")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
            ax.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="s")
            ax.set_title("Accuracy over Epochs", fontsize=14, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Evaluate
            loss, acc = model.evaluate(X_test_pad, y_test_cat, verbose=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Test Accuracy", f"{acc:.4f}")
            with col2:
                st.metric("📉 Test Loss", f"{loss:.4f}")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            y_pred = model.predict(X_test_pad, verbose=0)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test_cat, axis=1)

            cm = confusion_matrix(y_true, y_pred_labels)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Purples", fmt="d",
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        cbar_kws={"label": "Count"})
            ax2.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

            # Classification Report
            st.subheader("📄 Classification Report")
            report = classification_report(y_true, y_pred_labels, target_names=le.classes_)
            st.text(report)

# -------------------------------
# PREDICT NEWS
# -------------------------------
elif menu == "🔮 Predict News":
    st.title("🔮 Prediksi Kategori Berita")

    if "model" not in st.session_state:
        st.warning("⚠️ Silakan train model terlebih dahulu di menu 'Train Model'.")
    else:
        model = st.session_state["model"]
        tokenizer = st.session_state["tokenizer"]
        le = st.session_state["label_encoder"]

        st.write("### 📝 Masukkan Teks Berita")
        user_input = st.text_area(
            "Ketik atau paste teks berita di sini:",
            height=200,
            placeholder="Contoh: The government announced new policies today..."
        )

        if st.button("🔍 Prediksi Kategori", type="primary"):
            if user_input.strip() == "":
                st.error("❌ Teks tidak boleh kosong!")
            else:
                with st.spinner("🔄 Memproses..."):
                    clean = clean_text(user_input)
                    seq = tokenizer.texts_to_sequences([clean])
                    pad = pad_sequences(seq, maxlen=300, padding="post")

                    pred = model.predict(pad, verbose=0)
                    label = le.inverse_transform([np.argmax(pred)])
                    confidence = np.max(pred) * 100

                st.success(f"✅ Prediksi Selesai!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📰 Kategori Berita", label[0].upper())
                with col2:
                    st.metric("🎯 Confidence", f"{confidence:.2f}%")
                
                # Tampilkan probabilitas semua kelas
                st.subheader("📊 Probabilitas Semua Kategori")
                prob_df = pd.DataFrame({
                    "Kategori": le.classes_,
                    "Probabilitas (%)": pred[0] * 100
                }).sort_values("Probabilitas (%)", ascending=False)
                
                st.dataframe(prob_df, use_container_width=True)