import streamlit as st
import pandas as pd
import numpy as np

# import visualization package
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    # df = df.iloc[:,1:]
    return df


def run_eda_app():
    st.subheader("From Exploratory Data Analysis")
    df = load_data("human_capital.csv")
    df_survey = load_data("data_survey.csv")
    df_results = load_data("df_results.csv")
    # st.dataframe(df)

    submenu = st.sidebar.selectbox("SubMenu",["Description","Plots","Data Prediction"])
    if submenu == "Description":

        with st.expander("Dataset"):
            st.dataframe(df_survey)
        
        with st.expander("Descriptive Summary"):
            st.dataframe(df_survey.describe())

        with st.expander("Apakah Customer pernah menggunakan freelancer sebelumnya?"):
            st.dataframe(df_survey["Apakah Anda pernah menggunakan freelancer (penyedia jasa) sebelumnya?"].value_counts())


        with st.expander("Platform Freelancer apa yang sering digunakan?"):
            st.dataframe(df_survey["Platform freelancer"].value_counts())

        with st.expander("Jasa freelancer apa yang paling sering digunakan?"):
            st.dataframe(df_survey["Jasa freelancer apa yang paling sering Anda gunakan?"].value_counts())

    elif submenu == "Plots":
        st.subheader("Plots")

        # layouts
        col1,col2 = st.columns([2,1])

        with col1:
            with st.expander("Dist Plot of Platform Freelancer"):
                plat_df = df_survey['Platform freelancer'].value_counts().to_frame()
                plat_df = plat_df.reset_index()
                plat_df.columns = ['Platform', 'Counts']

                p1 = px.pie(plat_df, names='Platform', values='Counts')
                st.plotly_chart(p1, use_container_width=True)

        with col2:
            with st.expander("Platform Freelancer Distribution"):
                st.dataframe(plat_df)

        
        with st.expander("1. Hubungan Preferensi Platform dengan Age Range"):
                fig = plt.figure(figsize=(12, 6))
                sns.countplot(x='Platform freelancer', hue="Age Range", data=df_survey)
                plt.title('Hubungan Preferensi Platform dengan Age Range')
                plt.xlabel('Platform Freelancer')
                plt.ylabel('Jumlah Responden')
                plt.xticks(rotation=45)
                plt.legend(title='Age Range', bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)

        with st.expander("2. Hubungan Preferensi Platform dengan Jasa yang Digunakan"):
            fig = plt.figure(figsize=(12, 6))
            exclude_na = df_survey[df_survey['Jasa freelancer apa yang paling sering Anda gunakan?'] != 'NA']
            sns.countplot(x='Platform freelancer', hue='Jasa freelancer apa yang paling sering Anda gunakan?', data=exclude_na)
            plt.title('Hubungan Preferensi Platform dengan Jasa yang Digunakan')
            plt.xlabel('Platform Freelancer')
            plt.ylabel('Jumlah Responden')
            plt.xticks(rotation=45)
            plt.legend(title='Jasa Freelancer', bbox_to_anchor=(1.05, 1), loc='upper left')
            # fig = plt.show()
            st.pyplot(fig)

        with st.expander("3. Hubungan Preferensi Platform dengan SES Grade"):
            fig = plt.figure(figsize=(12, 6))
            sns.countplot(x='Platform freelancer', hue='SES Grade', data=df_survey)
            plt.title('Hubungan Preferensi Platform dengan SES Grade')
            plt.xlabel('Platform Freelancer')
            plt.ylabel('Jumlah Responden')
            plt.xticks(rotation=45)
            plt.legend(title='SES Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
        
        with st.expander("4. Hubungan Preferensi Platform dengan Job Status"):
            fig = plt.figure(figsize=(12, 6))
            sns.countplot(x='Platform freelancer', hue='Job Status', data=df_survey)
            plt.title('Hubungan Preferensi Platform dengan Job Status')
            plt.xlabel('Platform Freelancer')
            plt.ylabel('Jumlah Responden')
            plt.xticks(rotation=45)
            plt.legend(title='Job Status', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
        
        with st.expander("Correlation Plot"):
            columns = df_survey.select_dtypes(include='number')
            corr_matrix = columns.corr()
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
            st.pyplot(fig)
    elif submenu == "Data Prediction":
        st.text("Tabel data prediksi pengguna Sribu dengan probabilitas lebih dari 70%.")
        show_high_probability_data()
        # show_potential_customers()
    else:
        st.write("Gorila Coklat")


def show_high_probability_data():
    df = load_data("df_results.csv")
    
    # Allow user to select a threshold
    threshold = st.slider("Select Probability Threshold", 0.0, 1.0, 0.8)
    
    # Filter based on the selected threshold
    high_prob_data = df[df['Probability'] > threshold]
    
    # Show the filtered high-probability data
    st.subheader("Data with High Prediction Probabilities")
    st.dataframe(high_prob_data)

    non_users_df = df[df['Platform freelancer1'] == 0]
    potential_customers = non_users_df[non_users_df['Prediction'] == 1]
    
    # Display the filtered table
    st.subheader("Potential Customers for Sribu")
    st.dataframe(potential_customers)
