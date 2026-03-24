from anthropic import Anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import io
import os
import traceback

app = Flask(__name__)
CORS(app, origins="*")
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30.0)

def _message_text(message):
    parts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts).strip()

# Anthropic client — reads ANTHROPIC_API_KEY from environment automatically

# ── PDF text extraction ──────────────────────────────────────────────────────
def extract_pdf_text(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages[:40]:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        return ""

# ── Lesson generation ────────────────────────────────────────────────────────
def generate_lesson(text, topic_name="this topic", module_outline=None):
    system = "You are an expert university tutor for Money, Banking and Finance at the University of Birmingham. You create clear, accurate, exam-focused lessons from lecture notes."
    if module_outline:
        system += f"\n\nModule outline for context:\n{module_outline}"

    prompt = (
        f"Create a structured lesson on '{topic_name}' using these lecture notes:\n\n"
        f"{text[:4000]}\n\n"
        "Return ONLY a valid JSON object, no markdown, no backticks:\n"
        '{"title":"...","key_concepts":["concept 1","concept 2","concept 3"],'
        '"slides":[{"title":"slide title","body":"2-3 sentence explanation","highlight":"key formula or takeaway"}],'
        '"exam_tips":["tip 1","tip 2"]}'
        "\n\nInclude exactly 4 slides. Use the lecturer's notation where present."
    )

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1400,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = _message_text(message)
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    return json.loads(raw)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({"status": "StudyAI backend running"})

@app.route("/healthz")
def health():
    return jsonify({"status": "ok"})

@app.route("/debug_api", methods=["GET"])
def debug_api():
    """Small connectivity/auth test for Anthropic."""
    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with exactly: hello"}]
        )
        return jsonify({
            "ok": True,
            "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
            "response": _message_text(message)
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
            "error": str(e),
            "type": type(e).__name__,
            "trace": traceback.format_exc()
        }), 500

@app.route("/extract", methods=["POST"])
def extract():
    """Extract text from an uploaded PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = file.read()

    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    text = extract_pdf_text(file_bytes)

    if not text or len(text.strip()) < 50:
        return jsonify({"error": "Could not extract text from this PDF. Try saving as .txt instead."}), 422

    return jsonify({
        "text": text,
        "words": len(text.split()),
        "pages": len(text.split("\n"))
    })

@app.route("/lesson", methods=["POST"])
def lesson():
    """Generate a structured lesson from text using Claude."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    topic_name = data.get("topic", "this topic")
    module_outline = data.get("outline", None)

    if len(text.strip()) < 50:
        return jsonify({"error": "Text too short to generate a lesson"}), 400

    try:
        lesson_data = generate_lesson(text, topic_name, module_outline)
        return jsonify(lesson_data)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Claude returned invalid JSON: " + str(e),
            "type": type(e).__name__
        }), 500
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
            "trace": traceback.format_exc()
        }), 500

@app.route("/quiz", methods=["POST"])
def quiz():
    """Generate quiz questions from lesson content using Claude."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    topic_name = data.get("topic", "this topic")

    prompt = (
        f"Create 4 multiple choice quiz questions testing understanding of '{topic_name}'.\n\n"
        f"Content:\n{text[:3000]}\n\n"
        "Return ONLY a valid JSON array, no markdown, no backticks:\n"
        '[{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"...","concept":"2-4 word concept"}]'
        "\n\ncorrect is the 0-based index of the right answer."
    )

    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = _message_text(message)
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        questions = json.loads(raw)
        return jsonify(questions)
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/progress", methods=["GET"])
def get_progress():
    """Return saved progress for a user."""
    key = request.args.get("key", "default")
    path = f"/tmp/progress_{_safe_key(key)}.json"
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route("/progress", methods=["POST"])
def save_progress():
    """Save progress for a user."""
    key = request.args.get("key", "default")
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    path = f"/tmp/progress_{_safe_key(key)}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return jsonify({"saved": True})

def _safe_key(key):
    import hashlib
    return hashlib.md5(key.encode()).hexdigest()[:16]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)