from dotenv import load_dotenv
import os 
import streamlit as st
import pandas as pd
import numpy as np
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from bs4 import BeautifulSoup

load_dotenv()
API_KEY = os.environ['OPEN_AI_APIKEY']

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)



st.title("WebApp Integration of PandasAI for Prompt-Driven Analysis")
uploaded_file = st.file_uploader("Upload File For Analysis", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

prompt = st.text_area("Enter Your Prompt/Question:")

if st.button("Generate Answer"):
    if prompt:
        with st.spinner("Generating Response ..."):
         st.write(pandas_ai.run(df,prompt=prompt))
    
    else:
        st.warning("Please Enter Prompt/Question.")
        

        
            
