import json
from bs4 import BeautifulSoup
from tqdm import tqdm

def is_skip_data(flag, data):
    if flag == 'test':
        if data['annotations'][0]['yes_no_answer'] != "NONE":
            return True
        elif data['annotations'][0]['short_answers']:
            return True
    else:
        if data['annotations'][0]['yes_no_answer'] != "NONE":
            return True
        elif data['annotations'][0]['short_answers']:
            return True
    return False

def preprocess(batch, flag):
    contexts = []
    questions = []
    skipped_count = 0
    
    for example in batch:
        question = example['question_text']
        
        if is_skip_data(flag, example):
            context = ""
            skipped_count += 1
        else:
            long_answer = example['annotations'][0]['long_answer']
            start_byte = long_answer['start_byte']
            end_byte = long_answer['end_byte']
            
            with_html = example['document_html'][start_byte:end_byte]
            sliced_soup = BeautifulSoup(with_html, 'html.parser')
            context = sliced_soup.get_text(separator=' ')
        
        contexts.append(context)
        questions.append(question)
    
    return contexts, questions, skipped_count

def process_jsonl(input_file, output_file, flag, batch_size=1000):
    total_skipped = 0
    total_processed = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)

        batch = []
        for line in tqdm(f_in, total=total_lines, desc="Processing JSONL"):
            example = json.loads(line)
            batch.append(example)
            
            if len(batch) == batch_size:
                contexts, questions, skipped_count = preprocess(batch, flag)
                for context, question in zip(contexts, questions):
                    new_example = {"context": context, "question": question}
                    f_out.write(json.dumps(new_example) + '\n')
                total_skipped += skipped_count
                total_processed += len(batch)
                batch = []
                
        if batch:
            contexts, questions, skipped_count = preprocess(batch, flag)
            for context, question in zip(contexts, questions):
                new_example = {"context": context, "question": question}
                f_out.write(json.dumps(new_example) + '\n')
            total_skipped += skipped_count
            total_processed += len(batch)

    print(f"Total examples processed: {total_processed}")
    print(f"Total examples skipped: {total_skipped}")
    print(f"Percentage skipped: {(total_skipped / total_processed) * 100:.2f}%")
    

input_file = 'combined_dev.jsonl'
output_file = 'validation.jsonl'
flag = 'val'
process_jsonl(input_file, output_file, flag)