import streamlit as st
import streamlit.components.v1 as stc

from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#4DA1A9;padding:5px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Survey Market Sribu</h1>
		    </div>
            """

desc_temp = """
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """

def main():

    stc.html(html_temp)
    
    menu = ['Home', 'Exploratory Data Analysis','Machine Learning',]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()
    elif choice == "Exploratory Data Analysis":
        run_eda_app()


if __name__ == '__main__':
    main()
