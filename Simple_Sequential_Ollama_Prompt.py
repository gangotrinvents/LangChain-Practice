from langchain.llms import Ollama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

# Streamlit app setup
st.title("Testing Ollama")
input_text = st.text_input("Ask your question...")

# Initialize the OLLama LLM
llm = Ollama(model = 'phi', temperature = 0.8)

# Save information in memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Prompt templates for each chain. These will be used to interact with the LLM.
# Prompt1
input_prompt1 = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"  
)
chain1 = LLMChain(llm = llm, prompt = input_prompt1, verbose = True, output_key = 'person', memory = person_memory)

# Prompt2
input_prompt2 = PromptTemplate(
    input_variables=['name'],
    template="{name} born on"
)
chain2 = LLMChain(llm = llm, prompt = input_prompt2, verbose = True, output_key = 'dob', memory = dob_memory)

# Prompt3
input_prompt3 = PromptTemplate(
    input_variables=['dob'],
    template="Major events around the world on {dob}"
)
chain3 = LLMChain(llm = llm, prompt = input_prompt3, verbose = True, output_key = 'description', memory = descr_memory)

# Sequential chain to interact with the LLM.
overall_chain = SequentialChain(chains = [chain1, chain2, chain3], input_variables = ['name'],
                                 output_variables = ['person','dob','description'], verbose = True)

# Display the output in Streamlit.
if input_text:
    st.write(overall_chain({'name':input_text}))

    # Display the output saved in memory
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Description'):
        st.info(descr_memory.buffer)