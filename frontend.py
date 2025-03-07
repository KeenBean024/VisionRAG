# frontend.py
import streamlit as st
import requests

st.title("ColQwen2 RAG System")
pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
query = st.text_input("Enter your question")

if pdf_file:
    if st.button("Process Document"):
        requests.post("http://backend:5000/upload", files={'pdf': pdf_file})
        st.success("Document processed!")

if query and pdf_file:
    response = requests.post("http://backend:5000/query", json={
        'filename': pdf_file.name,
        'query': query
    })
    st.markdown(f"**Answer:** {response.json()['answer']}")
