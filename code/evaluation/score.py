import json
import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def compute_f1(prediction, ground_truth):
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)

    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()

    print(f"\nPrediction: {prediction}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Pred Tokens: {pred_tokens}")
    print(f"GT Tokens: {gt_tokens}")

    if len(gt_tokens) == 0:
        if len(pred_tokens) == 0:
            return 1.0 
        else:
            return 1.0

    common_tokens = set(pred_tokens) & set(gt_tokens)

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def compute_exact_match(prediction, ground_truth):
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    return int(prediction == ground_truth)

def load_validation_data(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def load_answers(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def evaluate_metrics(validation_data, answer_file):
    answers = load_answers(answer_file)
    
    f1_scores = []
    em_scores = []
    for example in validation_data:
        question = example['question']
        ground_truth = example['context']
        
        answer_entry = next((item for item in answers if item['question'] == question), None)
        
        if answer_entry:
            prediction = answer_entry['answers']
            f1_scores.append(compute_f1(prediction, ground_truth))
            em_scores.append(compute_exact_match(prediction, ground_truth))
        else:
            print(f"Answer not found for question: {question}")
            f1_scores.append(0.0)
            em_scores.append(0.0)

    average_f1 = sum(f1_scores) / len(f1_scores)
    average_em = sum(em_scores) / len(em_scores)
    print(f"Average F1 Score: {average_f1}")
    print(f"Average Exact Match Score: {average_em}")

if __name__ == "__main__":
    validation_file = '../data/validation.jsonl'
    validation_data = load_validation_data(validation_file)
    answer_file = '../results/answers_with_dual_gemini.jsonl'

    evaluate_metrics(validation_data, answer_file)