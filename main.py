# Integrate OpenAI API with LangChain

import os
import time
from constants import openai_key
from langchain_community.llms import OpenAI  # Updated import
import streamlit as st
import openai  # Import OpenAI to catch specific exceptions

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('LangChain Demo With OpenAI API')
input_text = st.text_input("Search the topic you want")

# OpenAI LLM
llm = OpenAI(temperature=0.8)

# Retry logic for API calls
def call_openai_with_retries(prompt, retries=5, backoff_factor=2):
    """
    Call the OpenAI API with retry logic.
    
    Args:
        prompt (str): The input prompt for the LLM.
        retries (int): Number of retry attempts.
        backoff_factor (int): Backoff factor for exponential delay.

    Returns:
        str: The response from the OpenAI API.
    """
    for attempt in range(retries):
        try:
            # Make the OpenAI API call
            return llm.invoke(prompt)  # Updated to use .invoke()
        except openai.error.RateLimitError:
            if attempt < retries - 1:  # Retry if attempts are remaining
                wait_time = backoff_factor ** attempt
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Rate limit error. Please check your OpenAI plan and usage.")
                raise
        except Exception as e:
            if attempt < retries - 1:  # Retry for other transient errors
                wait_time = backoff_factor ** attempt
                st.warning(f"An error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error(f"An unexpected error occurred: {e}. Please try again later.")
                raise

# If input text is provided, make the API call
if input_text:
    try:
        # Use retry logic for OpenAI call
        response = call_openai_with_retries(input_text)
        # Display the response
        st.write(response)
    except Exception as error:
        st.error("Failed to fetch the response. Please try again.")
