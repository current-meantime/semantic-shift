import json
import os
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import argparse
import pickle
from nltk.tokenize import sent_tokenize

# Setup
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {DEVICE}")
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def read_corpus(file_path):
    ext = os.path.splitext(file_path)[-1]
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    texts.append(item["text"])
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "text" in item:
                        texts.append(item["text"])
            elif "text" in data:
                texts.append(data["text"])
    return texts

def tokenize_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def collect_target_words(corpus, min_freq=50, max_words=10000):
    word_counts = Counter()
    for text in tqdm(corpus, desc="Counting words"):
        tokens = tokenize_text(text)
        word_counts.update(tokens)
    return [w for w, c in word_counts.items() if c >= min_freq][:max_words]

def chunk_text(text, max_words=250):
    """Split text into chunks of ~max_words using sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sentence)
        current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_contextual_embeddings(corpus, target_word, max_instances=100):
    embeddings = []
    found = 0
    for doc in corpus:
        chunks = chunk_text(doc, max_words=250)
        for chunk in chunks:
            if target_word not in chunk.lower():
                continue
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            token_ids = inputs['input_ids'][0]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            for i, tok in enumerate(tokens):
                if target_word in tok:
                    embeddings.append(outputs.last_hidden_state[0][i].cpu().numpy())
                    found += 1
                    break  # One occurrence per chunk
            if found >= max_instances:
                return np.array(embeddings)
    return np.array(embeddings)


def cluster_embeddings(embeddings, max_k=5):
    if len(embeddings) < 2:
        return [(0, np.mean(embeddings, axis=0))] if len(embeddings) > 0 else []
    best_score = -1
    best_labels = None
    best_k = 1
    for k in range(2, min(max_k+1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_labels = kmeans.labels_
            best_k = k
    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(embeddings)
    centers = [(label, centroid) for label, centroid in enumerate(kmeans.cluster_centers_)]
    return centers

def main(args):
    corpus = read_corpus(args.input)
    target_words = collect_target_words(corpus, min_freq=args.min_freq, max_words=args.max_words)

    all_word_clusters = {}
    for word in tqdm(target_words, desc="Processing words"):
        emb = extract_contextual_embeddings(corpus, word, max_instances=args.max_instances)
        if len(emb) < 5:
            continue
        clusters = cluster_embeddings(emb, max_k=args.max_k)
        all_word_clusters[word] = [(label, centroid.tolist()) for label, centroid in clusters]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_word_clusters, f)
        
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(all_word_clusters, f)
    except Exception as e:
        print(f"Failed to save embeddings: {e}")
        # Optionally, save a backup
        with open("backup_embeddings.pkl", "wb") as f:
            pickle.dump(all_word_clusters, f)
        


if __name__ == "__main__":
    class Args:
        input = r"C:\Users\alemr\OneDrive\Dokumenty\My Repos\us-presidency-archive-scraping\parsing_output\2000.json"
        output = "output/2000_embeddings.pkl"
        min_freq = 50
        max_words = 5000
        max_instances = 100
        max_k = 5

    args = Args()
    main(args)

    
    
    
'''    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Path to save .pkl clustered embeddings")
    parser.add_argument("--min_freq", type=int, default=50)
    parser.add_argument("--max_words", type=int, default=10000)
    parser.add_argument("--max_instances", type=int, default=100)
    parser.add_argument("--max_k", type=int, default=5)
    args = parser.parse_args()
    main(args)
'''