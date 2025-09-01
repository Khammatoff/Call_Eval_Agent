from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import split_into_sentences, chunk_by_tokens, normalize_whitespace

DEFAULT_MODEL = "blanchefort/rubert-base-cased-sentiment"  # RU модель
LABEL_MAP = {0: "негативный", 1: "нейтральный", 2: "положительный"}

@dataclass
class SegmentResult:
    label: str
    probs: Dict[str, float]
    text_len: int

class SentimentAggregator:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, max_tokens: int = 384):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_tokens = max_tokens
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    @torch.inference_mode()
    def _predict_chunk(self, chunk: str) -> SegmentResult:
        enc = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
        out = self.model(**enc)
        logits = out.logits.squeeze(0).detach().cpu()
        probs = torch.softmax(logits, dim=-1).tolist()
        probs_dict = {"негативный": probs[0], "нейтральный": probs[1], "положительный": probs[2]}
        label_idx = int(torch.argmax(logits).item())
        label = LABEL_MAP[label_idx]
        return SegmentResult(label=label, probs=probs_dict, text_len=len(chunk))

    def analyze(self, transcript: str) -> dict:
        text = normalize_whitespace(transcript)
        if not text:
            return {
                "label": "нейтральный",
                "confidence": 0.0,
                "details": [],
                "probs": {"негативный": 0.0, "нейтральный": 1.0, "положительный": 0.0},
            }

        sentences = split_into_sentences(text)
        chunks = chunk_by_tokens(sentences, self.tokenizer, max_tokens=self.max_tokens)

        results: List[SegmentResult] = []
        for ch in chunks:
            results.append(self._predict_chunk(ch))

        total_len = sum(r.text_len for r in results) or 1
        agg_probs = {
            "негативный": sum(r.probs["негативный"] * r.text_len for r in results) / total_len,
            "нейтральный": sum(r.probs["нейтральный"] * r.text_len for r in results) / total_len,
            "положительный": sum(r.probs["положительный"] * r.text_len for r in results) / total_len,
        }
        final_label = max(agg_probs, key=agg_probs.get)
        confidence = float(agg_probs[final_label])

        return {
            "label": final_label,
            "confidence": confidence,
            "probs": agg_probs,
            "details": [{"label": r.label, "probs": r.probs, "len": r.text_len} for r in results],
            "chunks": len(chunks),
        }
