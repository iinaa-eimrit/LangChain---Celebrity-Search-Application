## Integrate OpenAI API
import os
from constants import openai_key
from langchain_community.llms import OpenAI  # Updated import
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Celebrity Search Results')
input_text = st.text_input("Search for a celebrity")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celebrity {name}."
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLM
llm = OpenAI(temperature=0.8)

# LLM Chains
chain = LLMChain(
    llm=llm, 
    prompt=first_input_prompt, 
    verbose=True, 
    output_key='person', 
    memory=person_memory
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

chain2 = LLMChain(
    llm=llm, 
    prompt=second_input_prompt, 
    verbose=True, 
    output_key='dob', 
    memory=dob_memory
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

chain3 = LLMChain(
    llm=llm, 
    prompt=third_input_prompt, 
    verbose=True, 
    output_key='description', 
    memory=descr_memory
)

# Sequential Chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], 
    input_variables=['name'], 
    output_variables=['person', 'dob', 'description'], 
    verbose=True
)

# Streamlit Interaction
if input_text:
    # Run the chain
    output = parent_chain({'name': input_text})
    st.write(output)

    # Expanders to show memory
    with st.expander('Person Information'): 
        st.info(person_memory.buffer)

    with st.expander('Birth Date and Description'): 
        st.info(descr_memory.buffer)
