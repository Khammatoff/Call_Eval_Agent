from flask import Flask, render_template, request, jsonify
from core.sentiment import SentimentAggregator
from core.recommend import Recommender

app = Flask(__name__)

# Загружаем модели один раз
sa = SentimentAggregator()
rec = Recommender(enable_llm=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        transcript = request.form.get("transcript", "").strip()
        if not transcript:
            return render_template("index.html", error="Введите текст звонка.")
        sent = sa.analyze(transcript)
        tips = rec.generate(transcript, sent["label"])
        return render_template(
            "index.html",
            transcript=transcript,
            sentiment=sent["label"],
            confidence=round(sent["confidence"], 2),
            probs={k: round(v, 2) for k, v in sent["probs"].items()},
            recommendations=tips
        )
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    data = request.get_json(force=True)
    transcript = data.get("transcript", "")
    sent = sa.analyze(transcript)
    tips = rec.generate(transcript, sent["label"])
    return jsonify({
        "sentiment": sent["label"],
        "confidence": sent["confidence"],
        "probs": sent["probs"],
        "chunks": sent["chunks"],
        "recommendations": tips
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
