import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and data
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('models/model.pkl')

@st.cache
def load_data():
    return pd.read_csv('data/vgsales.csv')

model = load_model()
data = load_data()

# Sidebar navigation
st.sidebar.title("Navigasi")
options = st.sidebar.radio("Pilih Halaman:", ['Home', 'Prediksi Penjualan'])

# Home Page
if options == 'Home':
    st.title("Aplikasi Analisis Penjualan Video Games")
    st.write("Data Penjualan:")
    st.write(data.head())

# Prediction Page
elif options == 'Prediksi Penjualan':
    st.title("Prediksi Penjualan Global")
    st.write("Masukkan data penjualan regional untuk memprediksi penjualan global.")
    
    # Input features
    na_sales = st.number_input("Penjualan di Amerika Utara (NA Sales)", min_value=0.0, step=0.01)
    eu_sales = st.number_input("Penjualan di Eropa (EU Sales)", min_value=0.0, step=0.01)
    jp_sales = st.number_input("Penjualan di Jepang (JP Sales)", min_value=0.0, step=0.01)
    other_sales = st.number_input("Penjualan di Wilayah Lain (Other Sales)", min_value=0.0, step=0.01)
    
    # Predict
    if st.button("Prediksi"):
        input_features = np.array([[na_sales, eu_sales, jp_sales, other_sales]])
        prediction = model.predict(input_features)
        st.success(f"Prediksi Penjualan Global: {prediction[0]:.2f} juta kopi")
