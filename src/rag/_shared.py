"""Shared model and index configuration."""

from functools import lru_cache

MODEL = "intfloat/multilingual-e5-large"
TABLE = "knowledge"
BATCH_SIZE = 64
INDEX_VERSION = f"{MODEL}:markdown-tokens-v3-fts"


@lru_cache(maxsize=1)
def model():
    """Load and cache the embedding model on the available device."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging

    logging.disable_progress_bar()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL, device=device)
