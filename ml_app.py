import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models, scalers, and encoders
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
encoder = joblib.load('encoder.pkl')  # Your label or one-hot encoder

# Helper function to ensure alignment of one-hot encoded columns
from sklearn.preprocessing import OneHotEncoder

# Ensure that input_data contains only categorical columns for encoding
def preprocess_input_data(input_data):
    # Ensure you select only categorical columns that need encoding
    categorical_columns = ['Gender', 'Domicile', 'Job Status', 
                           'Jenis bisnis apa yang Anda operasikan?', 
                           'Jasa freelancer apa yang paling sering Anda gunakan?', 
                           'Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?']
    # One-hot encode categorical columns
    input_data_encoded = encoder.transform(input_data[categorical_columns])
    
    # Convert the encoded result (usually sparse matrix) into a DataFrame
    encoded_df = pd.DataFrame(input_data_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the encoded DataFrame with the remaining columns (numeric features)
    input_data_processed = pd.concat([input_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Ensure the input_data matches the expected format and order
    expected_columns = ['nominal__Gender_Perempuan', 'nominal__Domicile_Banten', 
                        'nominal__Domicile_Bengkulu', 'nominal__Domicile_DI Yogyakarta', 
                        'nominal__Domicile_DKI Jakarta', 'nominal__Domicile_Jawa Barat', 
                        'nominal__Domicile_Jawa Tengah', 'nominal__Domicile_Jawa Timur', 
                        'nominal__Domicile_Kalimantan Selatan', 'nominal__Domicile_Kalimantan Timur', 
                        'nominal__Domicile_Kepulauan Riau', 'nominal__Domicile_Lampung', 
                        'nominal__Domicile_Maluku', 'nominal__Domicile_Nanggroe Aceh Darussalam', 
                        'nominal__Domicile_Nusa Tenggara Barat', 'nominal__Domicile_Nusa Tenggara Timur', 
                        'nominal__Domicile_Riau', 'nominal__Domicile_Sulawesi Selatan', 
                        'nominal__Domicile_Sulawesi Tengah', 'nominal__Domicile_Sulawesi Utara', 
                        'nominal__Domicile_Sumatera Barat', 'nominal__Domicile_Sumatera Selatan', 
                        'nominal__Domicile_Sumatera Utara', 'nominal__Job Status_Bekerja penuh waktu', 
                        'nominal__Job Status_Jenis pekerjaan yang dibayar lainnya', 
                        'nominal__Job Status_Mahasiswa/pelajar', 'nominal__Job Status_Pemilik usaha/Wiraswasta', 
                        'nominal__Job Status_Tenaga lepas (freelancer)', 'nominal__Job Status_Tidak bekerja', 
                        'nominal__Jenis bisnis apa yang Anda operasikan?_E-commerce', 'nominal__Jenis bisnis apa yang Anda operasikan?_Fashion', 
                        'nominal__Jenis bisnis apa yang Anda operasikan?_Kuliner', 'nominal__Jenis bisnis apa yang Anda operasikan?_Manufaktur', 
                        'nominal__Jenis bisnis apa yang Anda operasikan?_Non-profit', 'nominal__Jenis bisnis apa yang Anda operasikan?_Other', 
                        'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_Layanan SEO', 'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_Manajemen media sosial', 
                        'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_NA', 'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_Other', 
                        'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_Pengembangan website', 'nominal__Jasa freelancer apa yang paling sering Anda gunakan?_Penulisan konten', 
                        'nominal__Business Owned Key_Ya', 'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_Harian', 
                        'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_Mingguan', 'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_NA', 
                        'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_Tahunan', 
                        'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_Tidak menentu', 
                        'nominal__Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?_Triwulanan', 'remainder__Age Range', 
                        'remainder__SES Grade', 'remainder__Berapa pendapatan tahunan bisnis Anda?', 
                        'remainder__Berapa anggaran pemasaran bulanan Anda?', 'remainder__Berapa jumlah karyawan yang Anda miliki?', 
                        'remainder__Platform freelancer1']  # Update expected columns

    # Add missing columns as zeros if they are not in the input data
    missing_cols = set(expected_columns) - set(input_data_processed.columns)
    for col in missing_cols:
        input_data_processed[col] = 0

    # Reorder the columns to match the expected order
    input_data_processed = input_data_processed[expected_columns]

    return input_data_processed

# Function to run the app and make predictions
def run_ml_app():
    st.title("Freelancer Platform Prediction")

    gender = st.radio('Gender', ['Laki-Laki', 'Perempuan'])
    domicile = st.selectbox('Domicile', ['Sumatera Selatan','Sumatera Utara','Riau','Banten','DKI Jakarta','Jawa Barat', 'Jawa Tengah','DI Yogyakarta','Jawa Timur','Bali','Sulawesi Selatan','Kalimantan Selatan','Sumatera Barat','Sulawesi Tengah'])
    job_status = st.selectbox('Job Status', ['Bekerja penuh waktu (full-time), status kontrak', 
                                             'Bekerja penuh waktu (full-time), status permanen', 
                                             'Pemilik usaha/Wiraswasta', 'Bekerja paruh waktu (part-time)', 
                                             'Tidak bekerja (ibu rumah tangga)', 'Pelajar SMA/SMK sederajat', 
                                             'Tidak bekerja (sedang mencari pekerjaan)', 
                                             'Jenis pekerjaan yang dibayar lainnya', 'Pelajar SMP sederajat', 
                                             'Mahasiswa aktif', 'Mahasiswa cuti kuliah'])
    business_type = st.selectbox('Business Type', ['E-commerce', 'Berbasis layanan', 'Manufaktur', 'Kuliner', 'Fashion', 'Non-profit', 'Other'])
    freelancer_service = st.selectbox('Freelancer Service Used', ['Manajemen media sosial', 'Pengembangan website', 'Desain logo', 
                                    'Penulisan konten', 'Layanan SEO', 'Other'])
    platform_experience = st.selectbox('Experience with Freelancer Platform', ['Tahunan', 'Bulanan', 'Mingguan', 'Harian', 'Triwulanan', 'Tidak menentu'])

    # Input for ordinal features
    age_range = st.selectbox('Age Range', ['25-30', '31-35', '36-40', '41-45'])
    annual_income = st.selectbox('Annual Income of Business', ['Kurang dari Rp750 juta', 'Rp750 juta - Rp1.5 miliar', 'Rp1.5 miliar - Rp7.5 miliar', 'Lebih dari Rp15 miliar', 'Rp7.5 miliar - Rp15 miliar'])
    marketing_budget = st.selectbox('Monthly Marketing Budget', ['Kurang dari Rp7.500.000', 'Rp7.500.000 - Rp15.000.000', 'Rp15.000.000 - Rp75.000.000', 'Rp75.000.000 - Rp150.000.000', 'Lebih dari Rp150.000.000'])
    employee_count = st.selectbox('Number of Employees', ['Kurang dari 5', '6-10','11-20','21-50','51-100','Lebih dari 100'])

    # Prepare input data in DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Domicile': [domicile],
        'Job Status': [job_status],
        'Jenis bisnis apa yang Anda operasikan?': [business_type],
        'Jasa freelancer apa yang paling sering Anda gunakan?': [freelancer_service],
        'Seberapa sering Anda menggunakan layanan freelancer (penyedia jasa) untuk proyek Anda?': [platform_experience],
        'Age Range': [age_range],
        'Berapa pendapatan tahunan bisnis Anda?': [annual_income],
        'Berapa anggaran pemasaran bulanan Anda?': [marketing_budget],
        'Berapa jumlah karyawan yang Anda miliki?': [employee_count]
    })
    if st.button('Predict'):
        # Preprocess the input data
        input_data_processed = preprocess_input_data(input_data)

        # Scale the data
        input_data_scaled = scaler.transform(input_data_processed)

        # Apply PCA
        input_data_pca = pca.transform(input_data_scaled)

        # Predict using the trained model
        prediction = model.predict(input_data_pca)

        # Show prediction result
        if prediction[0] == 1:
            st.write("Customer is likely to use Sribu.")
        else:
            st.write("Customer is not likely to use Sribu.")

