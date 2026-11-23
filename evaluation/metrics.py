import math

def retrieval_metrics(predicted, relevant, k=10):
    """Binary relevance metrics; relevant is the complete set of annotated relevant chunk IDs."""
    relevant = set(relevant)
    top = predicted[:k]
    hits = [int(cid in relevant) for cid in top]
    recall = sum(hits) / len(relevant) if relevant else 1.0
    precision = sum(hits) / k
    first = next((i for i, h in enumerate(hits, 1) if h), None)
    mrr = 1 / first if first else 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / ideal if ideal else 1.0
    return {'recall_at_10': recall, 'precision_at_10': precision, 'mrr_at_10': mrr, 'ndcg_at_10': ndcg}
