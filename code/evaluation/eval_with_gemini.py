import os
import json
import faiss
import time
import numpy as np
import google.generativeai as genai
from tqdm import tqdm
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import asyncio
from collections import deque

MAX_RETRIES = 6
INITIAL_RETRY_DELAY = 1
GEMINI_REQUESTS_PER_MINUTE = 1000
MAX_BATCH_SIZE = 1
ANSWER_FILE = "answers_with_gemini.json"

def load_or_create_answers():
    if os.path.exists(ANSWER_FILE):
        try:
            with open(ANSWER_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"'{ANSWER_FILE}' is empty or invalid. Starting with a new answer set.")
            return {}
    else:
        print(f"'{ANSWER_FILE}' not found. Starting with a new answer set.")
        with open(ANSWER_FILE, 'w') as f: 
            json.dump({}, f) 
        return {}

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
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    return distances[0], indices[0]

def load_validation_data(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

class RateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()

    async def wait(self):
        current_time = time.time()

        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        if len(self.request_times) >= self.requests_per_minute:
            await asyncio.sleep(60 - (current_time - self.request_times[0]))

        self.request_times.append(time.time())

gemini_limiter = RateLimiter(GEMINI_REQUESTS_PER_MINUTE)

async def call_gemini(prompts, model):
    await gemini_limiter.wait()
    retry_delay = INITIAL_RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            responses = await asyncio.to_thread(model.generate_content, prompts, stream=False)
            
            valid_responses = []
            for response in responses:
                if response.text:
                    valid_responses.append(response.text.strip())
                else:
                    print(f"Warning: Empty response from Gemini. Prompt feedback: {response.prompt_feedback}")
                    valid_responses.append("") 
            
            return valid_responses

        except Exception as e:
            if "Invalid operation" in str(e) and "prompt was blocked" in str(e):
                print(f"Prompt was blocked by Gemini API. Error: {e}")
                print(f"Returning empty string for this prompt. Consider reviewing the prompt content.")
                return [""] * len(prompts)  
            elif "Rate limit" in str(e):
                print(f"Gemini rate limit exceeded. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Error calling Gemini API: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    print("Max retries reached. Giving up.")
    return [""] * len(prompts) 

async def process_batch(batch, index, line_offsets, corpus_path, embedding_model, gemini_model, existing_answers):
    questions = [example['question'] for example in batch]

    query_embeddings = embedding_model.encode(questions)

    all_relevant_snippets = []
    all_retrieved_contexts = [] 
    gemini_prompts = []

    for i, query_embedding in enumerate(query_embeddings):
        distances, indices = search_faiss_index(index, query_embedding, top_k=1)
        relevant_snippets = []
        retrieved_contexts = []

        for j in range(len(indices)):
            offset = line_offsets[indices[j]]
            context = load_jsonl_line_by_offset(corpus_path, offset)
            relevant_snippets.append(context)
            retrieved_contexts.append(context[:100] + "...") 

        all_relevant_snippets.append(relevant_snippets)
        all_retrieved_contexts.append(retrieved_contexts)

        gemini_prompt = f'''
        You are a helpful and concise assistant.
        Provide an answer based on the Context.
        If the context doesn't contain useful information, you must generate with your knowledge.
        DO NOT LEAVE IT BLANK.

        Context: {" ".join(relevant_snippets)}

        Question: {questions[i]}

        Answer:
        '''
        gemini_prompts.append(gemini_prompt)

    predictions = await call_gemini(gemini_prompts, gemini_model)
    
    for i, prediction in enumerate(predictions):
        print("\n" + "="*50)
        print(f"Question: {questions[i]}")
        print("-"*50)
        print(f"Retrieved context: {all_retrieved_contexts[i][0]}") 
        print("-"*50)
        print(f"Final answer from Gemini: {prediction}")
        print("="*50)

        existing_answers[questions[i]] = prediction

        with open(ANSWER_FILE, "w") as f:
            json.dump(existing_answers, f, indent=4) 

    return questions, predictions

async def generate_answers(validation_data, index_path: str, corpus_path: str, embedding_model, batch_size: int, existing_answers: dict):
    print(f"Total validation data: {len(validation_data)}")
    print(f"Existing answers: {len(existing_answers)}")
    index = load_faiss_index(index_path)
    line_offsets = build_line_offsets(corpus_path)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    for i in tqdm(range(0, len(validation_data), batch_size), desc="Generating Answers", unit="batch"):
        batch = validation_data[i:i + batch_size]

        questions_to_process = []
        for example in batch:
            if example["question"] not in existing_answers or existing_answers[example["question"]] is None:
                questions_to_process.append(example)

        if not questions_to_process:
            continue

        print(f"Processing batch {i//batch_size + 1}/{len(validation_data)//batch_size + 1}")
        print(f"Questions to process in this batch: {len(questions_to_process)}")

        tasks = [process_batch(questions_to_process, index, line_offsets, corpus_path, embedding_model, gemini_model, existing_answers)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    index_path = "../data/index.faiss"
    corpus_path = "../data/corpus.jsonl"

    validation_file = '../data/validation.jsonl'
    validation_data = load_validation_data(validation_file)

    existing_answers = load_or_create_answers()

    asyncio.run(generate_answers(validation_data, index_path, corpus_path, embedding_model, batch_size=MAX_BATCH_SIZE, existing_answers=existing_answers))