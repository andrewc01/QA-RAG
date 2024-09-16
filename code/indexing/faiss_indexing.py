from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import torch
from tqdm import tqdm 

model = SentenceTransformer('all-MiniLM-L6-v2')

contexts = []
questions = []

with open('train_processed.jsonl', 'r') as f:
    for line in tqdm(f, desc="Processing JSONL", unit=" lines"):
        item = json.loads(line)
        context = item['context']
        question = item['question']
        
        contexts.append(context)
        questions.append(question)


batch_size = 4096  
embeddings = []

for i in tqdm(range(0, len(contexts), batch_size), desc="Generating embeddings", unit=" batches"):
    batch_contexts = contexts[i:i+batch_size]
    batch_embeddings = model.encode(batch_contexts, batch_size=batch_size, convert_to_tensor=True, device='cuda')
    embeddings.append(batch_embeddings.cpu().numpy())  

embeddings = np.vstack(embeddings)

d = embeddings.shape[1]

if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)

else:
    index = faiss.IndexFlatL2(d)

index.add(embeddings)

cpu_index = faiss.index_gpu_to_cpu(index)
cpu_index_file = 'index.faiss'
faiss.write_index(cpu_index, cpu_index_file)

print(f"CPU index saved as {cpu_index_file}")

query = "benefits of colonial life for single celled organisms"
query_vector = model.encode([query], convert_to_tensor=True, device='cuda').cpu().numpy()

D, I = index.search(query_vector, k=5)

for idx in I[0]:
    print(contexts[idx])

selected_context = contexts[I[0][0]] 
gpt_input = f"Context: {selected_context}\nQuestion: {query}"