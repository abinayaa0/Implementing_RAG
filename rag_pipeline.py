#!/usr/bin/env python3
"""
Advanced RAG Pipeline Implementation: Hybrid Search, RRF, Cross-Encoder Reranking, and LLM Generation.

This script contains a complete, modular, and runnable RAG pipeline. It includes:
1. Sparse Retriever (custom SimpleBM25 implementation)
2. Dense Retriever (Sentence-Transformers + Cosine Similarity)
3. Reciprocal Rank Fusion (RRF) rank combiner
4. Cross-Encoder Reranker (HuggingFace cross-encoder)
5. Simulated LLM generation step

You can run this script directly. It will attempt to load 'flat_chunks.json' from the
workspace, fit the retrievers, and execute a demo query.
"""

import os
import re
import sys
import math
import json
from collections import Counter
import numpy as np

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Try importing google-genai for Gemini API support
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ==========================================
# 1. Sparse Retriever (Simple BM25)
# ==========================================

def tokenize(text):
    """
    Tokenizes text by lowercasing and extracting alphanumeric word tokens.
    
    Input:
        text (str): The raw text to tokenize.
    Output:
        list of str: List of clean token strings.
    """
    return re.findall(r'\w+', text.lower())

class SimpleBM25:
    """
    A lightweight, self-contained implementation of the BM25 retrieval algorithm.
    """
    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        """
        Initializes BM25 and builds indices from the corpus.
        
        Inputs:
            corpus_texts (list of str): List of document chunk texts.
            k1 (float): BM25 term frequency saturation parameter.
            b (float): BM25 document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.tokenized_corpus = [tokenize(text) for text in corpus_texts]
        self.avg_doc_len = sum(len(doc) for doc in self.tokenized_corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.idf = {}
        self._initialize()

    def _initialize(self):
        # Count document frequencies (number of documents containing each word)
        nd = {}
        for doc in self.tokenized_corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        
        # Calculate IDF values for each word
        for word, freq in nd.items():
            # Standard BM25 IDF formula with smoothing to avoid negative values
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens):
        """
        Computes BM25 score for every document in the corpus against the query tokens.
        
        Input:
            query_tokens (list of str): Tokenized query.
        Output:
            list of float: List of BM25 scores matching index-wise with the corpus.
        """
        scores = [0.0] * self.corpus_size
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for idx, frequencies in enumerate(self.doc_freqs):
                freq = frequencies.get(word, 0)
                doc_len = self.doc_lens[idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[idx] += idf_val * (numerator / denominator)
        return scores


# ==========================================
# 2. Dense Retriever
# ==========================================

class DenseRetriever:
    """
    Dense retriever using SentenceTransformers to generate text embeddings
    and NumPy for cosine similarity search.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the retriever and attempts to load the sentence transformer model.
        """
        self.model_name = model_name
        self.embeddings = None
        self.is_mock = False
        
        if not HAS_SENTENCE_TRANSFORMERS:
            print("Warning: 'sentence-transformers' library is not available. Falling back to mock embeddings.")
            self.is_mock = True
            return
            
        print(f"Loading dense embedding model '{model_name}'...")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Failed to load sentence-transformers model '{model_name}': {e}")
            print("Falling back to mock embeddings.")
            self.is_mock = True

    def fit(self, texts, cache_path='embeddings_cache.npy'):
        """
        Computes embeddings for all corpus texts, using a disk cache if available.
        
        Input:
            texts (list of str): Corpus text chunks to embed.
            cache_path (str): File path for caching embeddings.
        """
        if self.is_mock:
            # Generate deterministic mock embeddings using numpy
            print("Generating mock dense embeddings...")
            np.random.seed(42)
            self.embeddings = np.random.randn(len(texts), 384)
            # Normalize mock embeddings to unit length
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms
            return

        # Check if we can load cached embeddings
        if cache_path and os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                if cached.shape[0] == len(texts):
                    print(f"Loaded cached dense embeddings for {len(texts)} chunks from '{cache_path}'.")
                    self.embeddings = cached
                    return
                else:
                    print(f"Cache size mismatch (cache: {cached.shape[0]}, texts: {len(texts)}). Recomputing...")
            except Exception as e:
                print(f"Warning: Failed to load cached embeddings: {e}. Recomputing...")

        print(f"Generating dense embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Normalize embeddings to unit length for fast cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        if cache_path:
            try:
                np.save(cache_path, self.embeddings)
                print(f"Saved dense embeddings cache to '{cache_path}'.")
            except Exception as e:
                print(f"Warning: Failed to save embeddings cache: {e}")

    def search(self, query_text, top_k=10):
        """
        Encodes the query and returns indices and cosine similarity scores.
        
        Inputs:
            query_text (str): Query to search for.
            top_k (int): Number of top results to return.
        Output:
            list of tuple: List of (doc_index, score) sorted descending by similarity.
        """
        if self.is_mock:
            # Generate stable mock query embedding based on word hashes
            np.random.seed(abs(hash(query_text)) % (2**32))
            query_emb = np.random.randn(384)
            query_emb /= np.linalg.norm(query_emb)
        else:
            query_emb = self.model.encode(query_text, show_progress_bar=False, convert_to_numpy=True)
            q_norm = np.linalg.norm(query_emb)
            if q_norm > 0:
                query_emb = query_emb / q_norm
        
        # Calculate cosine similarities (dot product since both are unit-normalized)
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, float(similarities[idx])) for idx in top_indices]


# ==========================================
# 3. Hybrid Search
# ==========================================

def hybrid_search(query, sparse_retriever, dense_retriever, top_k=10):
    """
    Executes both sparse and dense retrieval on the input query.
    
    Inputs:
        query (str): The search query.
        sparse_retriever (SimpleBM25): Initialized sparse retriever.
        dense_retriever (DenseRetriever): Initialized dense retriever.
        top_k (int): Number of top results to fetch from each retriever.
        
    Outputs:
        tuple: (sparse_results, dense_results)
            where each is a list of (doc_index, score) sorted descending.
    """
    # 1. Sparse search
    query_tokens = tokenize(query)
    sparse_scores = sparse_retriever.get_scores(query_tokens)
    sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
    sparse_results = [(idx, float(sparse_scores[idx])) for idx in sparse_indices]
    
    # 2. Dense search
    dense_results = dense_retriever.search(query, top_k=top_k)
    
    return sparse_results, dense_results


# ==========================================
# 4. Reciprocal Rank Fusion (RRF)
# ==========================================

def apply_rrf(sparse_results, dense_results, k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to combine sparse and dense rankings.
    
    Formula: RRF_Score(d) = 1 / (k + rank_sparse(d)) + 1 / (k + rank_dense(d))
    
    Inputs:
        sparse_results (list): List of (doc_index, score) from sparse retrieval (sorted).
        dense_results (list): List of (doc_index, score) from dense retrieval (sorted).
        k (int): The RRF smoothing constant (default is 60).
        
    Outputs:
        list of tuple: Sorted list of (doc_index, rrf_score) descending.
    """
    rrf_scores = {}
    
    # Helper to process a ranked list
    def score_ranking(results):
        for rank, (doc_idx, _) in enumerate(results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
            
    score_ranking(sparse_results)
    score_ranking(dense_results)
    
    # Sort docs by their fused RRF score descending
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf


# ==========================================
# 5. Cross-Encoder Reranker
# ==========================================

def rerank_with_cross_encoder(query, chunks, top_n_results, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Reranks the top N candidate documents using a HuggingFace Cross-Encoder model.
    
    Inputs:
        query (str): The search query.
        chunks (list of dict): List of chunk dictionaries containing 'text' keys.
        top_n_results (list of tuple): List of (doc_index, rrf_score) from RRF.
        model_name (str): The Cross-Encoder model to load.
        
    Outputs:
        list of tuple: Sorted list of (doc_index, score) descending from Cross-Encoder.
    """
    if not top_n_results:
        return []
        
    doc_indices = [idx for idx, _ in top_n_results]
    pairs = [(query, chunks[idx]['text']) for idx in doc_indices]
    
    # Check if sentence_transformers cross-encoder library is imported
    if not HAS_SENTENCE_TRANSFORMERS:
        print("Warning: Cross-Encoder library is unavailable. Performing rule-based mock reranking.")
        return _mock_reranking(query, chunks, doc_indices)
        
    print(f"Loading Cross-Encoder model '{model_name}'...")
    try:
        model = CrossEncoder(model_name)
        scores = model.predict(pairs)
        # Ensure scores is a list of floats
        scores = [float(s) for s in scores]
    except Exception as e:
        print(f"Warning: Failed to load Cross-Encoder model '{model_name}': {e}")
        print("Falling back to rule-based mock reranking.")
        return _mock_reranking(query, chunks, doc_indices)
        
    # Combine doc index and score, and sort
    reranked = list(zip(doc_indices, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked

def _mock_reranking(query, chunks, doc_indices):
    """
    A deterministic rule-based mock reranker that calculates a simple score
    based on exact term overlap and keyword matches to mock the reranking step.
    """
    scores = []
    query_words = set(tokenize(query))
    for idx in doc_indices:
        text = chunks[idx]['text']
        doc_words = set(tokenize(text))
        overlap = len(query_words.intersection(doc_words))
        
        # Simple scoring heuristic: overlap ratio + text length penalty/bonus
        ratio = float(overlap) / max(len(query_words), 1)
        # Add a tiny amount of noise so scores are not identical
        mock_score = ratio + 0.05 * math.sin(idx)
        scores.append(mock_score)
        
    reranked = list(zip(doc_indices, scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ==========================================
# 6. LLM Generation
# ==========================================

def format_context(chunks, reranked_results, top_k=3):
    """
    Formats the top K chunks into a single readable context block.
    
    Inputs:
        chunks (list of dict): List of chunk dictionaries containing 'text' keys.
        reranked_results (list of tuple): List of (doc_index, rerank_score).
        top_k (int): Number of top documents to put in the context.
        
    Output:
        str: Formatted context block.
    """
    formatted_docs = []
    for rank, (doc_idx, _) in enumerate(reranked_results[:top_k], start=1):
        chunk = chunks[doc_idx]
        text = chunk['text'].strip()
        # Include metadata if present in chunk
        meta_info = f"[Doc Index: {doc_idx}"
        if 'chunk_id' in chunk:
            meta_info += f" | ID: {chunk['chunk_id']}"
        if 'chapter_title' in chunk and chunk['chapter_title']:
            meta_info += f" | Chapter: {chunk['chapter_title']}"
        meta_info += "]"
        
        formatted_docs.append(f"--- Document {rank} {meta_info} ---\n{text}")
        
    return "\n\n".join(formatted_docs)

def generate_llm_response(query, context, model_name='gemini-2.5-flash'):
    """
    Generates a response from the LLM based on the user query and the retrieved context.
    If the GEMINI_API_KEY environment variable is set and the google-genai library is installed,
    it calls the real Google Gemini API. Otherwise, it falls back to a simulated response.
    
    Inputs:
        query (str): The search query.
        context (str): The retrieved context chunks.
        model_name (str): The Gemini model to use (default: gemini-2.5-flash).
        
    Outputs:
        str: Response text.
    """
    system_prompt = (
        "You are an advanced RAG assistant. Your task is to answer the user query based "
        "solely on the provided context. If the context does not contain enough information "
        "to answer the query, explain what is missing. Do not make up facts."
    )
    
    # Check if Gemini is available and API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    use_gemini = HAS_GEMINI and api_key
    
    # We display the final prompt constructed for inspection
    print("\n" + "="*40)
    if use_gemini:
        print(f"PROMPT SUBMITTED TO GEMINI ({model_name})")
    else:
        print("PROMPT SUBMITTED TO LLM (Simulated)")
    print("="*40)
    print(f"SYSTEM: {system_prompt}\n\nCONTEXT:\n{context}\n\nUSER QUERY: {query}")
    print("="*40 + "\n")
    
    if use_gemini:
        print(f"Calling Google Gemini API ({model_name})...")
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=f"Context:\n{context}\n\nQuery: {query}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            print(f"Warning: Failed to call Gemini API: {e}")
            print("Falling back to simulated response.")
    else:
        if not HAS_GEMINI:
            print("Notice: 'google-genai' package is not installed. Using simulated generation.")
        elif not api_key:
            print("Notice: GEMINI_API_KEY environment variable is not set. Using simulated generation.")
            print("To run with the real Gemini API, set the GEMINI_API_KEY environment variable.")

    # Fallback simulated response
    sentences = re.split(r'(?<=[.!?])\s+', context)
    query_tokens = [tok for tok in tokenize(query) if len(tok) > 3]
    
    relevant_sentences = []
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(token in s_lower for token in query_tokens):
            clean_s = sentence.replace('\n', ' ').strip()
            if len(clean_s) > 20 and clean_s not in relevant_sentences:
                relevant_sentences.append(clean_s)
                
    if relevant_sentences:
        answer_body = " ".join(relevant_sentences[:3])
        response = (
            f"According to the provided documents, we find relevant information matching your query.\n"
            f"Key details: {answer_body}\n"
            f"This summary is synthesized directly from the retrieved context pages (Simulated Response)."
        )
    else:
        response = (
            f"Based on the retrieved context, there are no specific details regarding '{query}'.\n"
            f"However, the document corpus covers standard procurement procedures, tender notices, "
            f"and works management guidelines (Simulated Response)."
        )
        
    return response



# ==========================================
# 7. Executable Demo Runner
# ==========================================

def load_document_chunks():
    """
    Loads chunks from 'flat_chunks.json' if it exists.
    If not found, loads a fallback list of high-quality mock chunks.
    
    Output:
        list of dict: List of document chunks.
    """
    chunks_path = 'flat_chunks.json'
    if os.path.exists(chunks_path):
        print(f"Loading actual document chunks from '{chunks_path}'...")
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print(f"Successfully loaded {len(data)} chunks.")
                    return data
        except Exception as e:
            print(f"Error loading {chunks_path}: {e}")
            
    # Fallback mock chunks
    print("Creating mock document chunks for the demo...")
    mock_chunks = [
        {
            "chunk_id": "chunk_0",
            "chapter_title": "Introduction to Works Procurement",
            "text": "The procurement of works in public systems follows strict guidelines. A preparation phase involves drawing up complete plans, designs, estimates, and obtaining administrative approval before any tender is invited."
        },
        {
            "chunk_id": "chunk_1",
            "chapter_title": "Tender Document Preparation",
            "text": "Tender documents must specify eligibility criteria, scope of work, technical specifications, and evaluation criteria. Pre-qualification criteria must be fair, transparent, and non-restrictive to promote healthy competition."
        },
        {
            "chunk_id": "chunk_2",
            "chapter_title": "E-Procurement and Portal Submission",
            "text": "All tenders above a threshold limit must be published on the Central Public Procurement Portal. Bidders submit their bids online, and the opening of bids is performed electronically by a designated committee."
        },
        {
            "chunk_id": "chunk_3",
            "chapter_title": "Bid Evaluation Guidelines",
            "text": "Bids are evaluated in two stages: technical evaluation and financial evaluation. The contract is normally awarded to the lowest responsive bidder (L1) whose bid meets the technical and financial specifications."
        },
        {
            "chunk_id": "chunk_4",
            "chapter_title": "Contract Management and Quality Control",
            "text": "Contract management requires regular monitoring of milestones. Payments are released against measurement books (MB) after physical verification of works. Quality control audits are carried out by third-party agencies."
        }
    ]
    return mock_chunks


def main():
    print("="*60)
    print("         ADVANCED RAG PIPELINE DEMO RUNNER")
    print("="*60)
    
    # 1. Load document corpus
    chunks = load_document_chunks()
    
    # To keep dense retrieval fast in a CPU demo, limit to the first 5000 chunks
    # if using the full actual dataset. You can change this limit as desired.
    MAX_DEMO_CHUNKS = 5000
    if len(chunks) > MAX_DEMO_CHUNKS:
        print(f"Corpus size is large ({len(chunks)} chunks).")
        print(f"Limiting to first {MAX_DEMO_CHUNKS} chunks for fast embedding generation in this demo.")
        chunks = chunks[:MAX_DEMO_CHUNKS]
        
    corpus_texts = [chunk['text'] for chunk in chunks]
    
    # 2. Initialize Retrievers
    # Sparse Retriever
    print("Initializing SimpleBM25 Sparse Retriever...")
    sparse_retriever = SimpleBM25(corpus_texts)
    
    # Dense Retriever
    dense_retriever = DenseRetriever()
    dense_retriever.fit(corpus_texts)
    
    # 3. Define Demo Query
    # Try using a query relevant to both actual procurement manual and mock data
    query ="What is the mathematical formula used for calculating the 'Available bid capacity' during the Pre-qualification Bidding (PQB) process?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"\nUser Query: '{query}'")
    
    # 4. Perform Hybrid Search
    print("\n[Step 1] Running Hybrid Search...")
    sparse_res, dense_res = hybrid_search(query, sparse_retriever, dense_retriever, top_k=10)
    
    print("\n--- Top Sparse (BM25) Results ---")
    for i, (idx, score) in enumerate(sparse_res[:5], start=1):
        print(f"{i}. Doc Index: {idx} | Score: {score:.4f} | Preview: {corpus_texts[idx][:90].strip()}...")
        
    print("\n--- Top Dense (Cosine Sim) Results ---")
    for i, (idx, score) in enumerate(dense_res[:5], start=1):
        print(f"{i}. Doc Index: {idx} | Score: {score:.4f} | Preview: {corpus_texts[idx][:90].strip()}...")
        
    # 5. Apply Reciprocal Rank Fusion (RRF)
    print("\n[Step 2] Applying Reciprocal Rank Fusion (RRF)...")
    rrf_res = apply_rrf(sparse_res, dense_res, k=60)
    
    print("\n--- Top Fused RRF Results (k=60) ---")
    for i, (idx, score) in enumerate(rrf_res[:5], start=1):
        print(f"{i}. Doc Index: {idx} | Fused Score: {score:.6f} | Preview: {corpus_texts[idx][:90].strip()}...")
        
    # 6. Apply Cross-Encoder Reranking
    # Take top N from RRF (e.g., top 5) and rerank
    top_n = min(5, len(rrf_res))
    candidates = rrf_res[:top_n]
    print(f"\n[Step 3] Reranking Top {top_n} candidates with Cross-Encoder...")
    
    reranked_res = rerank_with_cross_encoder(query, chunks, candidates)
    
    print("\n--- Top Reranked Results (Cross-Encoder) ---")
    for i, (idx, score) in enumerate(reranked_res, start=1):
        print(f"{i}. Doc Index: {idx} | Rerank Score: {score:.4f} | Preview: {corpus_texts[idx][:90].strip()}...")
        
    # 7. LLM Generation Context Preparation
    print("\n[Step 4] Formatting Top K chunks into Prompt Context...")
    top_k = min(3, len(reranked_res))
    context = format_context(chunks, reranked_res, top_k=top_k)
    
    # 8. LLM Generation
    print("\n[Step 5] Triggering Gemini LLM Response...")
    llm_response = generate_llm_response(query, context)
    
    print("\n--- LLM RESPONSE ---")
    print(llm_response)
    print("="*60)


if __name__ == '__main__':
    main()
