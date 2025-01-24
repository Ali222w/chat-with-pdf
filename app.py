import os
import tempfile
import streamlit as st
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def vector_embedding(uploaded_files):
    """
    Generate embeddings for uploaded PDF files and save them.
    
    Args:
        uploaded_files (list): List of uploaded PDF files
    
    Returns:
        str: Path to the saved embedding file
    """
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    all_docs = []
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load the PDF document
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()  # Load document content
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        # Add loaded documents to the list
        all_docs.extend(docs)
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(all_docs)
    
    # Create a vector store with FAISS
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    
    # Generate embedding file path
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    embedding_filename = f"{output_dir}/pdf_embeddings.faiss"
    
    # Save embeddings
    faiss.write_index(vectorstore.index, embedding_filename)
    
    # Save metadata and save additional supporting files
    with open(f"{embedding_filename}_metadata.pkl", "wb") as f:
        import pickle
        pickle.dump({
            "index_to_docstore_id": vectorstore.index_to_docstore_id,
            "docstore": vectorstore.docstore
        }, f)
    
    return embedding_filename

def main():
    st.title("PDF Embedding Generator")
    
    # Add Google API Key input
    google_api_key = st.text_input("Enter Google API Key", type="password")
    
    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Generate Embeddings"):
            try:
                # Set Google API Key
                os.environ["GOOGLE_API_KEY"] = google_api_key
                
                with st.spinner("Generating embeddings... Please wait."):
                    # Generate embeddings
                    embedding_file_path = vector_embedding(uploaded_files)
                    
                    # Provide download link
                    with open(embedding_file_path, "rb") as file:
                        st.download_button(
                            label="Download Embedding File",
                            data=file,
                            file_name="pdf_embeddings.faiss",
                            mime="application/octet-stream"
                        )
                    
                    # Also provide metadata download
                    with open(f"{embedding_file_path}_metadata.pkl", "rb") as file:
                        st.download_button(
                            label="Download Embedding Metadata",
                            data=file,
                            file_name="pdf_embeddings_metadata.pkl",
                            mime="application/octet-stream"
                        )
                    
                    st.success("Embeddings generated successfully! 🎉")
            
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
    
    st.sidebar.info("""
    ### How to Use
    1. Enter your Google API Key
    2. Upload one or more PDF files
    3. Click "Generate Embeddings"
    4. Download the generated embedding files
    """)

if __name__ == "__main__":
    main()
