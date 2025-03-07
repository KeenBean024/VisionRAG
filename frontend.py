import streamlit as st
import requests

st.title("ColQwen2 RAG System")

# Sidebar navigation for multi-view interface (as shown earlier)
view_option = st.sidebar.radio("Select Option", ("Upload PDF", "Query Document"))

if view_option == "Upload PDF":
    st.header("Upload PDF Document")
    pdf_file = st.file_uploader("Select a PDF file", type=['pdf'])
    if pdf_file is not None and st.button("Process Document"):
        try:
            response = requests.post("http://backend:5000/upload", files={'pdf': pdf_file})
            if response.ok:
                st.success("Document processed!")
            else:
                st.error("Failed to process document.")
        except Exception as e:
            st.error(f"Error: {e}")

elif view_option == "Query Document":
    st.header("Query a Document")
    try:
        resp = requests.get("http://backend:5000/filenames")
        filenames = resp.json().get("filenames", [])
    except Exception as e:
        st.error(f"Error fetching filenames: {e}")
        filenames = []
    
    if filenames:
        selected_filename = st.selectbox("Select a document", filenames)
        query = st.text_input("Enter your question")
        if st.button("Submit Query"):
            if query:
                try:
                    response = requests.post("http://backend:5000/query", json={
                        'filename': selected_filename,
                        'query': query
                    })
                    if response.ok:
                        data = response.json()
                        st.markdown(f"**Answer:** {data['answer']}")
                        for idx, b64_img in enumerate(data.get("pages", [])):
                            # Prepend the header for PNG images so st.image recognizes the format
                            image_data = f"data:image/png;base64,{b64_img}"
                            st.image(image_data, caption=f"Page {idx+1}")
                    else:
                        st.error("Query failed!")
                except Exception as e:
                    st.error(f"Error submitting query: {e}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("No documents available. Please upload a PDF first.")
