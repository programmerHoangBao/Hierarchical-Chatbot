import torch
import torch.nn.functional as F
import math
from collections import Counter

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def calculate_bleu(reference, candidate, n_max=4) -> float:
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    r = len(reference_tokens)
    c = len(candidate_tokens)
    if c == 0:
        return 0.0
    # Brevity penalty (BP)
    bp = 1 if c > r else math.exp(1 - (r/c))
    
    # Calculate the precision for each n-gram from 1 to n_max.
    p_n = []
    for n in range(1, n_max + 1):
        ref_ngrams = get_ngrams(reference_tokens, n)
        can_ngrams = get_ngrams(candidate_tokens, n)
        
        ref_counts = Counter(ref_ngrams)
        can_counts = Counter(can_ngrams)
        
        # Calculate the number of clipped matching n-grams.
        clipped_matches = 0
        for ngram, count in can_counts.items():
            clipped_matches += min(count, ref_counts[ngram])
        
        total_can_ngrams = len(can_ngrams)
        
        if total_can_ngrams == 0 or clipped_matches == 0:
            p_n.append(0.0)
        else:
            p_n.append(clipped_matches / total_can_ngrams)
    if 0.0 in p_n:
        return 0.0
    
    w = 1.0 / n_max
    log_p = sum(w * math.log(p) for p in p_n)
    bleu = bp * math.exp(log_p)
    return bleu

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_rouge_n(reference, candidate, n=1):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens or not cand_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        
    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)
    
    ref_counts = Counter(ref_ngrams)
    cand_counts = Counter(cand_ngrams)
    
    overlap = sum((ref_counts & cand_counts).values())
    
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
    f1 = calculate_f1(precision, recall)
    
    return {"recall": recall, "precision": precision, "f1": f1}

def calculate_lcs(ref_tokens, cand_tokens):
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def calculate_rouge_l(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if not ref_tokens or not cand_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        
    lcs_length = calculate_lcs(ref_tokens, cand_tokens)
    
    recall = lcs_length / len(ref_tokens)
    precision = lcs_length / len(cand_tokens)
    f1 = calculate_f1(precision, recall)
    
    return {"recall": recall, "precision": precision, "f1": f1}

def calculate_bertscore_matrix(
    model=None,
    tokenizer=None,
    reference=None,
    candidate=None
):

    if model is None or tokenizer is None or reference is None or candidate is None:
        raise ValueError("All parameters must be provided")

    model.eval()

    device = next(model.parameters()).device

    ref_inputs = tokenizer(
        reference,
        return_tensors="pt",
        truncation=True
    ).to(device)

    cand_inputs = tokenizer(
        candidate,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        ref_outputs = model(**ref_inputs)
        cand_outputs = model(**cand_inputs)

        ref_embs = ref_outputs.last_hidden_state[0, 1:-1, :]
        cand_embs = cand_outputs.last_hidden_state[0, 1:-1, :]

    if ref_embs.shape[0] == 0 or cand_embs.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "sim_matrix": None
        }

    ref_embs = F.normalize(ref_embs, p=2, dim=1)
    cand_embs = F.normalize(cand_embs, p=2, dim=1)

    sim_matrix = torch.matmul(cand_embs, ref_embs.T)

    precision_scores = sim_matrix.max(dim=1)[0]
    precision = precision_scores.mean().item()

    recall_scores = sim_matrix.max(dim=0)[0]
    recall = recall_scores.mean().item()

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sim_matrix": sim_matrix.cpu()
    }