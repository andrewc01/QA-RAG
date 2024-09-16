import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm
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

async def process_batch(batch, gemini_model, existing_answers):
    questions = [example['question'] for example in batch]

    gemini_prompts = []

    for question in questions:
        gemini_prompt = f'''
        You are a helpful and concise assistant.
        Provide an answer based on the Question.

        Question: {question}

        Answer:
        '''
        gemini_prompts.append(gemini_prompt)

    predictions = await call_gemini(gemini_prompts, gemini_model)
    
    for i, prediction in enumerate(predictions):
        print("\n" + "="*50)
        print(f"Question: {questions[i]}")
        print("-"*50)
        print(f"Answer from Gemini: {prediction}")
        print("="*50)

        existing_answers[questions[i]] = prediction

        with open(ANSWER_FILE, "w") as f:
            json.dump(existing_answers, f, indent=4) 

    return questions, predictions

async def generate_answers(validation_data, batch_size: int, existing_answers: dict):
    print(f"Total validation data: {len(validation_data)}")
    print(f"Existing answers: {len(existing_answers)}")

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

        tasks = [process_batch(questions_to_process, gemini_model, existing_answers)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    validation_file = '../data/validation.jsonl'
    validation_data = load_validation_data(validation_file)

    existing_answers = load_or_create_answers()

    asyncio.run(generate_answers(validation_data, batch_size=MAX_BATCH_SIZE, existing_answers=existing_answers))