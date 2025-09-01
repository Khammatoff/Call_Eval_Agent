import re
from typing import List

SENT_SEP_REGEX = re.compile(r'(?<=[.!?])\s+')

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = SENT_SEP_REGEX.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_by_tokens(sentences: List[str], tokenizer, max_tokens: int = 384) -> List[str]:
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_len = len(tokens)
        if sent_len > max_tokens - 2:
            tokens = tokens[:max_tokens - 2]
            sent = tokenizer.decode(tokens, skip_special_tokens=True)

        if current_len + sent_len <= max_tokens - 2:
            current.append(sent)
            current_len += sent_len
        else:
            if current:
                chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len

    if current:
        chunks.append(" ".join(current))
    if current:
        chunks.append(" ".join(current))
    return chunks
