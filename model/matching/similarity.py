"""
similarity.py
Functions for calculating similarity scores between queries and LLM-generated descriptions.
"""
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import numpy as np

from model.matching.clip_utils import clip_encode_text

def calculate_token_match_score(query: str, prompt: str) -> float:
    """Calculate token match score with fuzzy matching."""
    query = query.replace("-", " ")
    prompt = prompt.replace("color", "").replace("direction", "")
    stop_words = set(stopwords.words('english'))
    stop_words -= {"same"}
    stop_words |= {"color", "the", "which", "are", "of", "who", "a"}
    def tokenize_and_filter(sentence):
        tokens = word_tokenize(sentence.lower())
        return [token for token in tokens if token not in stop_words]
    tokens1 = tokenize_and_filter(query)
    tokens2 = tokenize_and_filter(prompt)
    def fuzzy_token_match(token1, token2):
        return SequenceMatcher(None, token1, token2).ratio()
    matched_tokens = set()
    overlap_score = 0
    for token1 in tokens1:
        best_match_score = 0
        best_match_token = None
        for token2 in tokens2:
            match_score = fuzzy_token_match(token1, token2)
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_token = token2
        if best_match_score > 0.8 and best_match_token not in matched_tokens:
            matched_tokens.add(best_match_token)
            overlap_score += 3 * best_match_score
    token_match_score = overlap_score / 10
    return token_match_score

def calculate_lidar_clip_similarity_score(query: str, answer: str, query_feat: torch.Tensor, clip_model, logit_scale_divisor: float = 100.0) -> float:
    """
    Calculate similarity score using LiDAR, CLIP, and TokenMatch.
    Args:
        query: str
        answer: str
        query_feat: torch.Tensor
        clip_model: CLIP model
        logit_scale_divisor: float, scaling factor (configurable)
    Returns:
        combined_score: float
    """
    llm_feat = clip_encode_text(answer, clip_model)
    llm_feat = llm_feat / llm_feat.norm(dim=-1, keepdim=True)
    query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.exp().float()
    logits_per_text = (logit_scale * llm_feat.float() @ query_feat.t().float()).item()
    token_match_score = calculate_token_match_score(query, answer)
    combined_score = logits_per_text / logit_scale_divisor + token_match_score
    return combined_score

