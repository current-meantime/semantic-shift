import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def load_clusters(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_shift(centroids1, centroids2):
    c1 = np.array([c for _, c in centroids1])
    c2 = np.array([c for _, c in centroids2])
    dist_matrix = cosine_distances(c1, c2)
    if dist_matrix.size == 0:
        return None
    min_dist = np.min(dist_matrix, axis=1)
    avg_shift = float(np.mean(min_dist))
    return avg_shift

def analyze_shift(clusters_old, clusters_new, shift_threshold=0.5):
    all_words = set(clusters_old) & set(clusters_new)
    shift_results = []

    for word in all_words:
        shift_score = compute_shift(clusters_old[word], clusters_new[word])
        if shift_score is not None:
            shift_results.append((word, shift_score))

    shift_results.sort(key=lambda x: -x[1])
    return shift_results

def print_report(results, top_n=30):
    print(f"\nTop {top_n} Semantic Shifts:\n")
    for word, score in results[:top_n]:
        print(f"{word:20} Shift Score: {score:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Path to 2000 clusters .pkl")
    parser.add_argument("--new", required=True, help="Path to 2024 clusters .pkl")
    args = parser.parse_args()

    old_clusters = load_clusters(args.old)
    new_clusters = load_clusters(args.new)

    results = analyze_shift(old_clusters, new_clusters)
    print_report(results)
