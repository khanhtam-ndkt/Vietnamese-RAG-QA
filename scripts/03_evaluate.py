import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import requests
from datasets import load_dataset
from tqdm import tqdm

from config import DATASET_NAME, EVAL_SPLIT, API_PORT, RERANK_TOP_K

API_URL = f"http://localhost:{API_PORT}/retrieve"
MAX_EVAL_SAMPLES = 500

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(any(rid in relevant_ids for rid in retrieved_ids[:k]))

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    import math
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rid in enumerate(retrieved_ids[:k], 1) if rid in relevant_ids)
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    print(f"Loading {DATASET_NAME} / {EVAL_SPLIT} …")
    ds = load_dataset(DATASET_NAME)[EVAL_SPLIT]
    
    # --- ADD FILTER HERE (MATCHES KAGGLE) ---
    print(f"Original dataset size: {len(ds)}")
    ds = ds.filter(lambda row: len(row["answers"]["text"]) > 0)
    print(f"Filtered dataset size (answerable only): {len(ds)}")

    samples = list(ds)
    if MAX_EVAL_SAMPLES:
        samples = samples[:MAX_EVAL_SAMPLES]
        print(f"Evaluating on {len(samples)} samples")

    recall_scores, ndcg_scores, failures = [], [], []
    failed = 0

    for row in tqdm(samples, desc="Evaluating"):
        question = row["question"]
        gt_context = row["context"].strip()
        gold_answers = [ans.lower().strip() for ans in row["answers"]["text"]]

        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            data = resp.json()
        except Exception as e:
            failed += 1
            continue

        passages = data.get("passages", [])
        retrieved_texts = [p["text"] for p in passages]

        def is_relevant(text: str, gt: str, threshold: float = 0.6) -> bool:
            text, gt = text.strip(), gt.strip()
            if not text: return False
            window = 40
            if len(text) < window: return text in gt
            hits = sum(1 for i in range(0, len(text) - window, window // 2) if text[i:i+window] in gt)
            total_windows = max(1, len(range(0, len(text) - window, window // 2)))
            return (hits / total_windows) >= threshold

        relevant_mask = []
        for text in retrieved_texts:
            # --- INCORPORATE EXACT ANSWER MATCH (MATCHES KAGGLE) ---
            if gold_answers and any(ans in text.lower() for ans in gold_answers):
                relevant_mask.append(True)
            elif is_relevant(text, gt_context):
                relevant_mask.append(True)
            else:
                relevant_mask.append(False)

        retrieved_ids = [str(i) for i in range(len(retrieved_texts))]
        relevant_ids  = {str(i) for i, m in enumerate(relevant_mask) if m}

        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, RERANK_TOP_K))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, RERANK_TOP_K))

        if not any(relevant_mask):
            failures.append({
                "question": question,
                "gold_answers": gold_answers,
                "ground_truth_context": gt_context,
                "top_3_retrieved": [f"[Rank {p.get('final_rank', i+1)}] {p.get('text', '')}" for i, p in enumerate(passages[:3])]
            })

    n = len(recall_scores)
    if n == 0:
        print("No successful evaluations. Check API is running.")
        return

    print(f"\n{'='*40}")
    print(f"Evaluation Results (n={n}, failed={failed})")
    print(f"{'='*40}")
    print(f"Recall@{RERANK_TOP_K}  : {sum(recall_scores)/n:.4f}  (target ≥ 0.9000)")
    print(f"nDCG@{RERANK_TOP_K}    : {sum(ndcg_scores)/n:.4f}  (target ≥ 0.8920)")

    os.makedirs("data", exist_ok=True)
    with open("data/failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=4)
        
    print(f"\nSaved {len(failures)} failed retrievals to data/failures.json")

if __name__ == "__main__":
    main()