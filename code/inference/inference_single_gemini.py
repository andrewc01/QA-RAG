import os
import json
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

def build_line_offsets(corpus_path: str) -> List[int]:
    print(f"Building line offsets for {corpus_path}")
    offsets = []
    with open(corpus_path, 'r') as f:
        offset = 0
        while line := f.readline():
            offsets.append(offset)
            offset += len(line)
    print(f"Built offsets for {len(offsets)} lines")
    return offsets

def load_faiss_index(index_path: str) -> faiss.Index:
    print(f"Loading Faiss index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Loaded Faiss index successfully")
    return index

def load_jsonl_line_by_offset(corpus_path: str, offset: int) -> str:
    with open(corpus_path, 'r') as f:
        f.seek(offset)
        line = f.readline()
        item = json.loads(line)
        return item['context']

def search_faiss_index(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]

def generate_answer_and_context(question: str, index_path: str, corpus_path: str, embedding_model, top_k: int = 5) -> Tuple[str, str]:
    index = load_faiss_index(index_path)
    line_offsets = build_line_offsets(corpus_path)

    query_embedding = embedding_model.encode([question])
    
    distances, indices = search_faiss_index(index, query_embedding, top_k)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat()

    contexts = []
    for idx in indices:
        offset = line_offsets[idx]
        context = load_jsonl_line_by_offset(corpus_path, offset)
        contexts.append(context)

    combined_context = "\n\n".join(contexts)

    answer_prompt = f'''
    You are a helpful and concise assistant.
    Provide an answer based on the Context.
    If no relevant information is available, generate an answer based on your knowledge.

    Context: {combined_context}

    Question: {question}

    Answer:
    '''

    response = chat.send_message(answer_prompt)
    answer = response.text.strip()

    return answer, combined_context

if __name__ == "__main__":
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    question = "What is Transformer model?"
    index_path = "../data/index.faiss"
    corpus_path = "../data/corpus.jsonl"

    # Call the function
    answer, retrieved_contexts = generate_answer_and_context(question, index_path, corpus_path, embedding_model, top_k=1)

    # Print results
    print('-'*100)
    print(f"{retrieved_contexts[:200]}")
    print('-'*100)
    print("Question:", question)
    print('-'*100)
    print("Answer:", answer)
    print('-'*100)