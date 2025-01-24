import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import pickle  # For saving embeddings
import replicate  # Library to interact with Replicate API

# Set the Replicate API key
os.environ["REPLICATE_API_TOKEN"] = "r8_Fe7wFx3APBz178VhZ9DVfHdCKcRDIrm0ADLrY"

# Function to generate embeddings using Replicate
def generate_embeddings_with_replicate(texts):
    embeddings = []
    model = "meta/llama-2-7b-chat"
    for text in texts:
        # Request embeddings from Replicate
        response = replicate.run(model, input={"text": text})
        embeddings.append(response)  # Append embedding for each chunk of text
    return embeddings

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

                # Prepare text for embedding
                texts = [doc.page_content for doc in final_documents]

                # Generate embeddings using Replicate
                st.info("Generating embeddings...")
                embeddings = generate_embeddings_with_replicate(texts)

                # Create FAISS vector store
                st.info("Saving embeddings to FAISS vector store...")
                vector_store = FAISS.from_texts(texts, embeddings)

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
