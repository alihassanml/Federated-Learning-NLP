"""
Utility functions for document processing and FAISS indexing
"""
import os
import PyPDF2
import docx
import faiss
import numpy as np
from typing import List, Dict
import json
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
        return ""

def load_documents_from_folder(folder_path):
    """
    Load all documents from a folder
    
    Returns:
        List of dicts with 'text' and 'source' keys
    """
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return documents
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        text = ""
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        
        if text.strip():
            documents.append({
                'text': text,
                'source': filename
            })
    
    print(f"Loaded {len(documents)} documents from {folder_path}")
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # At least 50% of chunk_size
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def create_faiss_index(embeddings, index_type='flat'):
    """
    Create FAISS index from embeddings
    
    Args:
        embeddings: numpy array of shape (n_docs, embedding_dim)
        index_type: 'flat' or 'ivf'
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    if index_type == 'flat':
        index = faiss.IndexFlatL2(dimension)
    else:  # IVF for larger datasets
        nlist = min(100, embeddings.shape[0] // 10)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, k=5):
    """
    Search FAISS index
    
    Args:
        index: FAISS index
        query_embedding: Query embedding (1D array)
        k: Number of results to return
    
    Returns:
        distances, indices
    """
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

def save_index_and_chunks(index, chunks, metadata, save_dir):
    """Save FAISS index and associated data"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(save_dir, 'faiss_index.bin'))
    
    # Save chunks
    with open(os.path.join(save_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index_and_chunks(load_dir):
    """Load FAISS index and associated data"""
    index = faiss.read_index(os.path.join(load_dir, 'faiss_index.bin'))
    
    with open(os.path.join(load_dir, 'chunks.json'), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    with open(os.path.join(load_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, chunks, metadata

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()