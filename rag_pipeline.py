"""
RAG Pipeline Implementation
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import RetrieverModel, GeneratorModel
from utils import *
import numpy as np
import os

class RAGDataset(Dataset):
    """Dataset for RAG training"""
    
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'context': self.contexts[idx],
            'answer': self.answers[idx]
        }

class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation"""
    
    def __init__(self, company_id, data_folder, retriever_model='sentence-transformers/all-mpnet-base-v2', 
                 generator_model='google/flan-t5-base',use_lora=True):
        self.company_id = company_id
        self.data_folder = data_folder
        self.retriever = RetrieverModel(retriever_model)
        self.generator = GeneratorModel(generator_model, use_lora=use_lora)
        self.use_lora = use_lora
        
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        print(f"RAG Pipeline initialized for {company_id}")
        print(f"  Retriever: {retriever_model}")
        print(f"  Generator: {generator_model}")
        print(f"  LoRA: {'Enabled' if use_lora else 'Disabled'}") 
    
    def load_and_index_documents(self, use_multimodal=True):
        # Load documents and create FAISS index
        
        # Args:
        #     use_multimodal: If True, use multimodal PDF parsing (tables, images, OCR)

        print(f"Loading documents from {self.data_folder}")
        print(f"Multimodal parsing: {'Enabled' if use_multimodal else 'Disabled'}")
        
        # Load documents with multimodal support
        documents = load_documents_from_folder(self.data_folder, use_multimodal=use_multimodal)

        
        if not documents:
            print("No documents found. Creating sample data.")
            self._create_sample_data()
            documents = load_documents_from_folder(self.data_folder)
        
        # Chunk documents
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            chunks = chunk_text(doc['text'], chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'source': doc['source'],
                        'chunk_id': i
                    })
        
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.retriever.encode(all_chunks)
        
        # Create FAISS index
        print("Building FAISS index...")
        self.index = create_faiss_index(embeddings)
        
        # Save index
        save_dir = f"models/{self.company_id}"
        save_index_and_chunks(self.index, self.chunks, self.chunk_metadata, save_dir)
        
        print(f"Indexing complete. Total chunks: {len(self.chunks)}")
        return len(self.chunks)
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant chunks for a query"""
        if self.index is None:
            raise ValueError("Index not created. Call load_and_index_documents() first.")
        
        # Encode query
        query_embedding = self.retriever.encode([query])[0]
        
        # Search
        distances, indices = search_faiss_index(self.index, query_embedding, k=top_k)
        
        # Get chunks
        retrieved_chunks = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.chunks):
                retrieved_chunks.append({
                    'text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'score': float(distance)
                })
        
        return retrieved_chunks
    
    def generate_answer(self, query, retrieved_chunks):
        """Generate answer from query and retrieved chunks"""
        # Combine context
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        # Create prompt
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate
        answer = self.generator.generate(prompt, max_length=128)
        
        return answer
    
    def query(self, question, top_k=3):
        """Complete RAG query pipeline"""
        # Retrieve
        retrieved_chunks = self.retrieve(question, top_k=top_k)
        
        # Generate
        answer = self.generate_answer(question, retrieved_chunks)
        
        return {
            'answer': answer,
            'sources': retrieved_chunks
        }
    
    def train_step(self, questions, answers, learning_rate, epochs, dp_noise=0.0, batch_size=16, round_number=0):
        """
        Training step for the generator model with optional Differential Privacy (DP)
        Args:
            questions (list[str]): List of questions
            answers (list[str]): List of answers
            learning_rate (float): Learning rate
            epochs (int): Number of epochs
            dp_noise (float): DP noise multiplier
            batch_size (int): Batch size
            round_number (int): Current federated round (for optional warm-up)
        Returns:
            avg_loss (float): Average loss over all epochs and batches
        """
        import random
        import torch

        self.generator.model.train()
        trainable_params = [p for p in self.generator.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        total_loss = 0
        num_batches_total = 0

        # OPTIONAL: warm-up rounds without DP
        if round_number < 3:
            dp_noise = 0.0

        for epoch in range(epochs):
            # Shuffle data each epoch
            combined = list(zip(questions, answers))
            random.shuffle(combined)
            questions_shuffled, answers_shuffled = zip(*combined)

            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(questions_shuffled), batch_size):
                batch_questions = questions_shuffled[i:i+batch_size]
                batch_answers = answers_shuffled[i:i+batch_size]

                # Retrieve contexts
                contexts = []
                for q in batch_questions:
                    try:
                        retrieved = self.retrieve(q, top_k=5)
                        context = " ".join([chunk["text"] for chunk in retrieved])
                        context = context[:800]  # max 800 chars
                    except:
                        context = ""
                    contexts.append(context)

                # Prepare inputs and labels
                inputs = [f"Context: {c}\n\nQuestion: {q}\n\nAnswer:" for q, c in zip(batch_questions, contexts)]
                input_encodings = self.generator.tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
                target_encodings = self.generator.tokenizer(batch_answers, padding=True, truncation=True, max_length=128, return_tensors="pt")

                input_encodings = {k: v.to(self.generator.device) for k, v in input_encodings.items()}
                labels = target_encodings["input_ids"].to(self.generator.device)
                labels[labels == self.generator.tokenizer.pad_token_id] = -100

                # Forward
                outputs = self.generator.model(**input_encodings, labels=labels)
                loss = outputs.loss

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # DP per-batch gradient noise
                if dp_noise > 0:
                    for p in trainable_params:
                        if p.grad is not None:
                            # gradient clipping
                            norm = torch.norm(p.grad)
                            clip_norm = 1.0
                            if norm > clip_norm:
                                p.grad *= (clip_norm / norm)
                            # add Gaussian noise
                            p.grad += torch.randn_like(p.grad) * dp_noise

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

            total_loss += epoch_loss
            num_batches_total += num_batches

        avg_loss = total_loss / num_batches_total if num_batches_total > 0 else 0
        print(f"Training completed. Overall Average Loss: {avg_loss:.4f}")
        return avg_loss



    
    def _create_sample_data(self):
        """Create sample documents if none exist"""
        os.makedirs(self.data_folder, exist_ok=True)
        
        if self.company_id == "company1":
            sample_doc = """
            Company Policy Document - HR Guidelines
            
            Employee Leave Policy:
            All full-time employees are entitled to 20 days of paid leave per year.
            Sick leave is separate and provides up to 10 days annually.
            
            Working Hours:
            Standard working hours are 9 AM to 5 PM, Monday through Friday.
            Flexible working arrangements can be requested through the HR portal.
            
            Remote Work Policy:
            Employees may work remotely up to 3 days per week with manager approval.
            All remote work must be logged in the timesheet system.
            """
        else:
            sample_doc = """
            Technical Documentation - IT Systems
            
            VPN Access:
            To access company resources remotely, use the Cisco AnyConnect VPN client.
            Credentials are the same as your email login.
            
            Password Policy:
            Passwords must be at least 12 characters long.
            Must include uppercase, lowercase, numbers, and special characters.
            Passwords expire every 90 days.
            
            Support Contact:
            For IT support, email support@company.com or call extension 5555.
            Emergency support is available 24/7.
            """
        
        with open(os.path.join(self.data_folder, 'sample_doc.txt'), 'w') as f:
            f.write(sample_doc)
        
        print(f"Created sample document for {self.company_id}")