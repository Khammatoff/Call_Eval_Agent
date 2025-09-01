from typing import List, Optional
from transformers import pipeline
import torch

GEN_MODEL = "google/flan-t5-small"

class Recommender:
    def __init__(self, enable_llm: bool = True, device: Optional[str] = None, max_new_tokens: int = 64):
        self.enable_llm = enable_llm
        self.max_new_tokens = max_new_tokens
        self.device = device

        if self.enable_llm:
            self.pipe = pipeline(
                "text2text-generation",
                model=GEN_MODEL,
                tokenizer=GEN_MODEL,
                device=0 if (device is None and torch.cuda.is_available()) else -1
            )
        else:
            self.pipe = None

    def _heuristic(self, transcript: str, sentiment: str) -> List[str]:
        tips = []
        lower = transcript.lower()

        if sentiment == "негативный":
            tips.append("Говорите спокойнее, подтверждайте, что услышали клиента.")
            tips.append("Задавайте уточняющие вопросы и предлагайте решение.")
        elif sentiment == "нейтральный":
            tips.append("Добавьте эмпатию и благодарность.")
            tips.append("Структурируйте ответ: резюме, шаги, сроки.")
        else:
            tips.append("Сохраните тон, подытоживайте договоренности.")
            tips.append("Уточните следующий шаг и ответственность.")

        if any(k in lower for k in ["перебива", "перебил", "перебиваете", "перебиваю"]):
            tips.append("Не перебивайте клиента, дождитесь паузы.")
        if any(k in lower for k in ["не слышно", "плохо слышно", "связь прерывается"]):
            tips.append("Проверьте качество связи и перефразируйте ключевые моменты.")
        if any(k in lower for k in ["дорого", "слишком дорого", "цена"]):
            tips.append("Объясните ценность и предложите альтернативы.")

        return tips[:3]

    def generate(self, transcript: str, sentiment: str) -> List[str]:
        prompt = (
            "Звонок (транскрипт):\n"
            f"{transcript[:1500]}\n\n"
            f"Тон разговора: {sentiment}.\n"
            "Дай 1–2 короткие рекомендации на русском, как улучшить этот разговор."
        )
        if self.pipe is None:
            return self._heuristic(transcript, sentiment)

        try:
            out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, num_return_sequences=1)
            text = out[0]["generated_text"].strip()
            parts = [p.strip("-• ").strip() for p in text.replace("\n", ". ").split(".") if p.strip()]
            parts = [p for p in parts if len(p) > 2][:2]
            if not parts:
                parts = self._heuristic(transcript, sentiment)
            return parts[:2]
        except Exception:
            return self._heuristic(transcript, sentiment)
