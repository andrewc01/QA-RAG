import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import hashlib

def preprocess(example):
    with_html = example['document_html']
    sliced_soup = BeautifulSoup(with_html, 'html.parser')
    context = sliced_soup.get_text(separator=' ')
    context = re.sub(r'Contents[^\n]*\n([0-9.\s\n]*)\n', '', context)
    context = ' '.join(context.split())

    question = example['question_text']
    
    return context, question

def get_context_hash(context):
    return hashlib.md5(context.encode('utf-8')).hexdigest()

def process_jsonl_file_consistent(input_file, output_file, batch_size=1000):
    processed_hashes = set()
    skipped_count = 0
    total_count = 0

    unique_examples = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        total_lines = 307373
        f_in.seek(0)
        for line in tqdm(f_in, total=total_lines, desc="Identifying unique contexts"):
            example = json.loads(line)
            context, question = preprocess(example)
            context_hash = get_context_hash(context)
            
            if context_hash not in processed_hashes:
                processed_hashes.add(context_hash)
                unique_examples.append({"context": context, "question": question})
            else:
                skipped_count += 1
            total_count += 1

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(unique_examples), batch_size):
            batch = unique_examples[i:i+batch_size]
            for example in batch:
                f_out.write(json.dumps(example) + '\n')

    print(f"Total examples processed: {total_count}")
    print(f"Total contexts skipped: {skipped_count}")
    print(f"Total unique contexts: {len(unique_examples)}")

input_file = 'combined_train.jsonl'
output_file = 'train_processed.jsonl'

if __name__ == '__main__':
    process_jsonl_file_consistent(input_file, output_file, batch_size=2000)