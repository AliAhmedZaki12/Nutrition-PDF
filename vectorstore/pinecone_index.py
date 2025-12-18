from pinecone import Pinecone
import streamlit as st

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

def get_index():
    return pc.Index(st.secrets["PINECONE_INDEX"])

