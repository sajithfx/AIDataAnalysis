import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Data Analysis",  # This sets the <title> in the browser tab
    page_icon="🧠",                 # Optional: adds an icon to the tab
    # layout="wide"                   # Optional: makes layout wider
)

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
else:
    st.title("AI Data Analysis")
    uploaded_file = st.file_uploader(
        "Choose a file", type=("csv", "xlsx", "xls"))

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("First 500 rows of the uploaded file")
        st.write(df.head())

        question = st.text_input("Enter the question about your data here")

        if question:
            def create_agent(dataframe):
                llm = OpenAI(api_key=openai_api_key)
                agent = create_pandas_dataframe_agent(
                    llm, dataframe, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True,)
                return agent

            agent = create_agent(df)
            with st.spinner("Analyzing..."):
                answer = agent.run(question)
            st.write("Answer:")
            st.write(answer)
    else:
        st.write("Please upload a CSV or Excel file to proceed.")
