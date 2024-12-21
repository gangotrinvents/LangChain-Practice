from langchain.llms import Ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Initialize the OLLama LLM
llm = Ollama(model = 'phi', temperature = 0.8)

# Initialize streamlit
st.title("Testing Ollama")
input_txt = st.text_input("Ask your question...")

if input_txt:
    st.write(llm(input_txt))