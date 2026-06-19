import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset
from tqdm import tqdm

from config import (
    DATASET_NAME,
    DATA_DIR,
    PROCESSED_CONTEXTS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_WORDS,
)

def split_into_chunks(text, chunk_size=80, overlap_size=30, min_chunk_words=20):
    """
    Chunks text strictly by word count using a sliding window.
    Guarantees exact overlaps and zero redundant micro-chunks.
    """
    words = text.split()
    
    # Calculate how many words to advance the window by
    # max(1, ...) ensures we always move forward even if config is misconfigured
    step = max(1, chunk_size - overlap_size) 
    
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        
        # Drop chunks that don't meet the minimum word threshold
        # (unless the text itself is tiny, in which case we take what we can get)
        if len(chunk_words) >= min_chunk_words or not chunks:
            chunks.append(" ".join(chunk_words))
            
    return chunks

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"[1/3] Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    print("[2/3] Extracting unique contexts and chunking …")
    seen_contexts = set()
    all_chunks = []
    chunk_id = 0

    for split_name in ["train", "validation"]:
        if split_name not in ds:
            continue
        for row in tqdm(ds[split_name], desc=split_name):
            ctx = row["context"].strip()
            if ctx in seen_contexts:
                continue
            seen_contexts.add(ctx)

            title = row.get("title", "")
            
            # Use the new word-level sliding window with config injection
            chunks = split_into_chunks(
                text=ctx, 
                chunk_size=CHUNK_SIZE, 
                overlap_size=CHUNK_OVERLAP, 
                min_chunk_words=MIN_CHUNK_WORDS
            )
            
            for chunk in chunks:
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "title": title,
                    "text": chunk,
                    "source_split": split_name,
                })
                chunk_id += 1

    print(f"    → {len(seen_contexts):,} unique contexts → {len(all_chunks):,} chunks")

    print(f"[3/3] Saving to {PROCESSED_CONTEXTS_PATH}")
    with open(PROCESSED_CONTEXTS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if all_chunks:
        avg_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
        print(f"    → avg chunk: {avg_len:.0f} chars")

if __name__ == "__main__":
    main()