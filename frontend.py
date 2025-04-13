import streamlit as st
import requests
import base64
from io import BytesIO

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="ColQwen2 RAG System")

# Custom CSS for better styling and full-width layout
st.markdown("""
<style>
    .st-emotion-cache-1jicfl2 {
        width: 100%;
        padding: 6rem 1rem 10rem;
        min-width: auto;
        max-width: initial;
    }
    .stApp {
        margin: 0 auto;
    }
    .step-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e1ecf4;
        border-left: 5px solid #2196F3;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main layout
left, main, right = st.columns([1,6,1])

with main:
    st.title("🤖 ColQwen2 RAG System")
    st.markdown("---")

    # Sidebar navigation for multi-view interface
    with st.sidebar:
        st.header("Navigation")
        view_option = st.radio("Select Option", ("Upload PDF", "Query Document", "Query with Image"))
        st.markdown("---")
        st.info("👋 Welcome to ColQwen2 RAG System! This tool allows you to upload PDFs, query documents, and even use images in your queries.")

    if view_option == "Upload PDF":
        st.header("📤 Upload PDF Document")
        
        with st.expander("ℹ️ How to upload a PDF", expanded=True):
            st.markdown("""
            1. Click on 'Browse files' or drag and drop your PDF.
            2. Once a file is selected, click 'Process Document'.
            3. Wait for the confirmation message.
            """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            pdf_file = st.file_uploader("Select a PDF file", type=['pdf'])
        with col2:
            process_button = st.button("Process Document", type="primary", disabled=not pdf_file)
        
        if pdf_file is not None and process_button:
            with st.spinner("Processing document..."):
                try:
                    response = requests.post("http://backend:5000/upload", files={'pdf': pdf_file})
                    if response.ok:
                        st.success("✅ Document processed successfully!")
                    else:
                        st.error("❌ Failed to process document.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    elif view_option == "Query Document":
        st.header("🔍 Query a Document")
        
        with st.expander("ℹ️ How to query a document", expanded=True):
            st.markdown("""
            1. Select a document from the dropdown.
            2. Enter your question in the text box.
            3. Click 'Submit Query' to get your answer.
            """)
        
        try:
            resp = requests.get("http://backend:5000/filenames")
            filenames = resp.json().get("filenames", [])
        except Exception as e:
            st.error(f"❌ Error fetching filenames: {e}")
            filenames = []
        
        if filenames:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_filename = st.selectbox("Select a document", filenames)
                query = st.text_input("Enter your question")
            with col2:
                submit_button = st.button("Submit Query", type="primary", disabled=not query)
            
            if submit_button:
                with st.spinner("Generating answer..."):
                    try:
                        response = requests.post("http://backend:5000/query", json={
                            'filename': selected_filename,
                            'query': query
                        })
                        if response.ok:
                            data = response.json()
                            st.markdown("### Answer")
                            st.info(data['answer'])
                            st.markdown("### Relevant Pages")
                            for idx, b64_img in zip(data.get("top_pages", []), data.get("pages", [])):
                                image_data = f"data:image/png;base64,{b64_img}"
                                st.image(image_data, caption=f"Page {idx+1}", use_container_width=True)
                        else:
                            st.error("❌ Query failed!")
                    except Exception as e:
                        st.error(f"❌ Error submitting query: {e}")
        else:
            st.info("📚 No documents available. Please upload a PDF first.")

    elif view_option == "Query with Image":
        st.header("🖼️ Query with Image")
        
        with st.expander("ℹ️ How to query with an image", expanded=True):
            st.markdown("""
            1. Select a document from the dropdown.
            2. Enter your question in the text box.
            3. Upload an image related to your query.
            4. Click 'Submit Query' to get your answer.
            """)
        
        try:
            resp = requests.get("http://backend:5000/filenames")
            filenames = resp.json().get("filenames", [])
        except Exception as e:
            st.error(f"❌ Error fetching filenames: {e}")
            filenames = []
        
        if filenames:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_filename = st.selectbox("Select a document", filenames)
                query = st.text_input("Enter your question")
            with col2:
                uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
                if uploaded_image:
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            
            submit_button = st.button("Submit Query", type="primary", disabled=not (query and uploaded_image))
            
            if submit_button:
                with st.spinner("Generating answer..."):
                    try:
                        img_bytes = uploaded_image.getvalue()
                        files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
                        data = {'filename': selected_filename, 'query': query}
                        
                        response = requests.post("http://backend:5000/query/image", 
                                                 files=files,
                                                 data=data)
                        
                        if response.ok:
                            data = response.json()
                            st.markdown("### Answer")
                            st.info(data['answer'])
                            st.markdown("### Relevant Pages")
                            for idx, b64_img in zip(data.get("top_pages", []), data.get("pages", [])):
                                image_data = f"data:image/png;base64,{b64_img}"
                                st.image(image_data, caption=f"Page {idx+1}", use_container_width=True)
                        else:
                            st.error("❌ Query failed!")
                    except Exception as e:
                        st.error(f"❌ Error submitting query: {e}")
        else:
            st.info("📚 No documents available. Please upload a PDF first.")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by KeenBean")
