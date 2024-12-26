from langchain.llms import Ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from langchain.chains import LLMChain
from langchain import PromptTemplate

input_prompt = PromptTemplate(
    template="Tell me about this celebrity {input_text}",
    input_variables=["input_text"]
)

# Initialize the OLLama LLM
llm = Ollama(model = 'phi', temperature = 0.8   )
chain = LLMChain(llm = llm, prompt = input_prompt)

# Initialize streamlit
st.title("Testing Ollama")
input_txt = st.text_input("Ask your question...")

if input_txt:
    st.write(chain.run(input_txt))