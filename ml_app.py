import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os

attribute_info = """
                 - Department: Sales & Marketing, Operations, Technology, Analytics, R&D, Procurement, Finance, HR, Legal
                 - Region: region 1 - region 34
                 - Educaiton: Below Secondary, Bachelor's, Master's & above
                 - Gender: Male and Female
                 - Recruitment Channel: Referred, Sourcing, Others
                 - No of Training: 1-10
                 - Age: 10-60
                 - Previous Year Rating: 1-5
                 - Length of Service: 1-37 Month
                 - Awards Won: 1. Yes, 0. No
                 - Avg Training Score: 0-100
                 """

age = {'25-30':1, '31-35':2, '36-40':3, '41-45':4}
gen = {'Laki-Laki':1, 'Perempuan':2}
ses = {'Upper':1, 'Middle':2, 'Lower':3}
bok = {'Ya':1, 'Tidak':2}
job = {
    'Bekerja penuh waktu (full-time), status kontrak':1,
    'Bekerja penuh waktu (full-time), status permanen':2,
    'Pemilik usaha/Wiraswasta':3, 
    'Bekerja paruh waktu (part-time)':4,
    'Tidak bekerja (ibu rumah tangga)':5,
    'Pelajar SMA/SMK sederajat':6,
    'Tidak bekerja (sedang mencari pekerjaan)':7, 
    'Jenis pekerjaan yang dibayar lainnya':8,
    'Pelajar SMP sederajat':9,
    'Mahasiswa aktif':10,
    'Mahasiswa cuti kuliah': 11
}
bisnisType = {'E-commerce':1, 'Berbasis layanan':2, 'Manufaktur':3, 'Kuliner':4, 'Fashion':5, 'Non-profit':6, 'Other':7}
jasa = {'Manajemen media sosial':1, 'Pengembangan website':2, 'Desain logo':3, 'Penulisan konten':4, 'Layanan SEO':5, 'Other':6}
platform = {'Other':1, 'Sribu':2}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    st.subheader("ML Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    range_age = st.selectbox('Range Age', ['25-30', '31-35', '36-40', '41-45'])
    gender = st.radio('Gender', ['Laki-Laki', 'Perempuan'])
    ses_grade = st.selectbox('SES Grade', ['Upper', 'Middle', 'Lower'])
    job_status = st.selectbox("Job Status", ['Bekerja penuh waktu (full-time), status kontrak', 
                                             'Bekerja penuh waktu (full-time), status permanen', 
                                             'Pemilik usaha/Wiraswasta', 'Bekerja paruh waktu (part-time)', 
                                             'Tidak bekerja (ibu rumah tangga)', 'Pelajar SMA/SMK sederajat', 
                                             'Tidak bekerja (sedang mencari pekerjaan)', 
                                             'Jenis pekerjaan yang dibayar lainnya', 'Pelajar SMP sederajat', 
                                             'Mahasiswa aktif', 'Mahasiswa cuti kuliah'])
    b_owned_key = st.selectbox('Business Owned Key', ['Ya', 'Tidak'])
    jenis_bisnis = st.selectbox('Jenis bisnis apa yang Anda operasikan?', ['E-commerce', 'Berbasis layanan', 
                                                                           'Manufaktur', 'Kuliner', 'Fashion', 
                                                                           'Non-profit', 'Other'])
    jasa_freelance = st.selectbox('Jasa freelancer apa yang paling sering Anda gunakan?', 
                                   ['Manajemen media sosial', 'Pengembangan website', 'Desain logo', 
                                    'Penulisan konten', 'Layanan SEO', 'Other'])
    platform_freelace = st.selectbox('Platform freelancer (penyedia jasa) mana yang paling sering Anda gunakan?', 
                                      ['Other', 'Sribu'])

    # Encode Input
    result = {
        'Age Range': range_age,
        'Gender': gender,
        'SES Grade': ses_grade,
        'Job Status': job_status,
        'Business Owned Key': b_owned_key,
        'Jenis bisnis apa yang Anda operasikan?': jenis_bisnis,
        'Jasa freelancer apa yang paling sering Anda gunakan?': jasa_freelance,
        'Platform freelancer1': platform_freelace,
    }

    # Encode Values Directly
    encoded_result = [
        get_value(result['Age Range'], age),
        get_value(result['Gender'], gen),
        get_value(result['SES Grade'], ses),
        get_value(result['Job Status'], job),
        get_value(result['Business Owned Key'], bok),
        get_value(result['Jenis bisnis apa yang Anda operasikan?'], bisnisType),
        get_value(result['Jasa freelancer apa yang paling sering Anda gunakan?'], jasa),
        get_value(result['Platform freelancer1'], platform),
    ]
    expected_columns = ['Age Range', 'Gender', 'SES Grade', 'Job Status', 'Business Owned Key', 
                    'Jenis bisnis apa yang Anda operasikan?', 'Jasa freelancer apa yang paling sering Anda gunakan?', 
                    'Platform freelancer1']

    # prediction section
    st.subheader('Prediction Result')
    # Load the scaler and model
    model = load_model("model_tuned.pkl")
    scaler = load_model("scaler.pkl")

    X_test = pd.DataFrame([encoded_result], columns=expected_columns)

    # Scale Data
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)




    # Perform PCA on the scaled data
    pca = load_model("pca.pkl")
    X_test_pca = pca.transform(X_test_scaled)

    # Melakukan prediksi
    prediction = model.predict(X_test_pca)
    prediction_prob = model.predict_proba(X_test_pca)[:,1]

    # Menampilkan hasil prediksi
    if prediction == 0:
        st.write("Hasil Prediksi: Kelas 0 (tidak berpotensi) - Tidak ada risiko.")
    else:
        st.write("Hasil Prediksi: Kelas 1 (berpotensi) - Ada risiko.")
    
    # Menampilkan probabilitas prediksi untuk Kelas 0 dan Kelas 1
    st.write(f"Probabilitas Kelas 0: {prediction_prob[0][0]:.4f}")
    st.write(f"Probabilitas Kelas 1: {prediction_prob[0][1]:.4f}")

    # Menampilkan hasil prediksi dengan visualisasi (optional)
    st.subheader("Visualisasi Prediksi")
    if prediction == 0:
        st.markdown("<h3 style='color:green;'>Model Menyatakan Tidak Ada Risiko</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red;'>Model Menyatakan Ada Risiko</h3>", unsafe_allow_html=True)

import sklearn
print(sklearn.__version__)