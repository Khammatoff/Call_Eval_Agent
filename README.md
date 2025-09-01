
# Агент оценки звонков (Flask + PyTorch + RU-сентимент)

Минимальный агент, который принимает **транскрипт звонка** на русском языке и возвращает:
- **Тон:** положительный / нейтральный / негативный
- **1–2 короткие рекомендации**, как улучшить разговор

## Стек технологий
- **Python**, **Flask** — веб-интерфейс + REST API
- **PyTorch**, **Transformers (Hugging Face)** — локальные модели
- **Модель сентимента (RU):** [`blanchefort/rubert-base-cased-sentiment`](https://huggingface.co/blanchefort/rubert-base-cased-sentiment) (3 класса: негативный, нейтральный, положительный)
- **Модель рекомендаций:** [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) (или эвристический фолбек)

## Структура проекта

```
call-eval-agent/
├─ app.py                  # Flask UI + REST API
├─ core/
│  ├─ sentiment.py         # RU-сентимент + разбиение на блоки + агрегация
│  ├─ recommend.py         # Генерация рекомендаций
│  └─ utils.py             # Вспомогательные функции
├─ templates/index.html    # HTML-шаблон интерфейса
├─ requirements.txt
├─ Dockerfile
├─ README.md
├─ diagram.png
└─ ARCHITECTURE.md
```

---

## Запуск локально

### 1) Установка

```bash

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Запуск Flask

```bash

python app.py
```

Приложение доступно на `http://localhost:5000`.

---

## REST API

Для интеграции с n8n или другими системами:

- **POST** `http://localhost:5000/api/analyze`
- **JSON Body:**
```json
{
    "transcript": "Добрый день! ..."
}
```

- **Пример ответа JSON:**
```json
{
    "sentiment": "положительный",
    "confidence": 0.87,
    "probs": {"негативный": 0.05, "нейтральный": 0.08, "положительный": 0.87},
    "chunks": 2,
    "recommendations": [
        "Сохраните тон, подытоживайте договоренности.",
        "Уточните следующий шаг и ответственность."
    ]
}
```

---

## Docker

### Сборка образа

```bash

docker build -t call-eval-agent:latest .
```

### Запуск приложения

```bash

docker run --rm -p 5000:5000 call-eval-agent:latest
```

---

## Кастомизация

- **Отключить генеративные рекомендации:**  
  В `app.py` создайте `Recommender(enable_llm=False)`.
- **Сменить модель сентимента:**  
  В `core/sentiment.py` поменяйте `DEFAULT_MODEL` на другую RU-модель Hugging Face.
- **Метод агрегации:**  
  Сейчас используется **взвешенное среднее** вероятностей по длине текста; можно заменить на **медиану**, **максимум** или **голосование**.

---

## Ограничения

- Модель `blanchefort/rubert-base-cased-sentiment` хорошо работает на русском, но специфическая лексика звонков может определяться не идеально.
- `flan-t5-small` — маленькая LLM, рекомендации общие; для более качественных советов можно заменить на большую инструкционную модель.
- Разбиение текста на предложения выполнено через простой regex; можно улучшить с помощью `razdel` или `spacy`.

---

## Что можно улучшить при большем времени

- Fine-tune RU-сентимент на своих звонках.
- Мультиспикерный анализ (кто говорил, скорость речи, длительность пауз).
- Дашборд с метриками звонков и рекомендациями оператору.
- Более точное распознавание эмоций и перебиваний.

---

## Ссылки на модели

- RU сентимент: [`blanchefort/rubert-base-cased-sentiment`](https://huggingface.co/blanchefort/rubert-base-cased-sentiment)
- Генеративные рекомендации: [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)
