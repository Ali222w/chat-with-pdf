import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import pickle  # For saving embeddings

# Initialize API key variables
google_api_key = None

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    st.write("Enter your Google API key to proceed.")
    google_api_key = st.text_input("Enter your Google API key:", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

# Main area
st.title("PDF Embedding Generator")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    if st.button("Generate Embeddings"):
        with st.spinner("Processing the PDF..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                # Load the PDF document
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()

                # Split documents into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                final_documents = text_splitter.split_documents(docs)

                # Generate embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
                vector_store = FAISS.from_documents(final_documents, embeddings)

                # Save the FAISS vector store to a file
                embedding_file = "pdf_embeddings.pkl"
                with open(embedding_file, "wb") as f:
                    pickle.dump(vector_store, f)

                st.success(f"Embeddings saved to {embedding_file} successfully!")

                # Provide a download link for the file
                st.download_button(
                    label="Download Embedding File",
                    data=open(embedding_file, "rb").read(),
                    file_name=embedding_file,
                    mime="application/octet-stream",
                )
            finally:
                # Remove the temporary file
                os.remove(temp_file_path)
