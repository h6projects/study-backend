from anthropic import Anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import io
import os
import re
import traceback
import psycopg2
import psycopg2.extras

app = Flask(__name__)
CORS(app, origins="*")
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30.0)

def _get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
    conn.commit()
    return conn

TOPIC_CONTEXT = {
    "Overview of the Financial System & Interest Rates": "financial system, flow of funds, financial markets, financial intermediaries, interest rates, meaning of interest rates, nominal vs real rates",
    "Measures of Risk & Behaviour of Interest Rates": "risk measurement, standard deviation, variance, bond prices, loanable funds, liquidity preference, Fisher effect",
    "Term Structure of Interest Rates & Stock Market": "yield curve, expectations theory, liquidity premium, stock market, equity valuation, market microstructure",
    "Efficient Market Hypothesis & Derivative Markets": "EMH, weak strong semi-strong efficiency, options, futures, forwards, derivatives, arbitrage",
    "Behavioural Finance": "investor psychology, biases, heuristics, overconfidence, loss aversion, market anomalies, irrational behaviour",
    "Principles of Finance & Return and Risk Measurement": "principles of finance, financial system, return measurement, risk measurement, time value, cash flows",
    "Time Value of Money": "present value, future value, discounting, compounding, annuity, perpetuity, NPV basics, interest rate, time value",
    "Discounted Cash Flow Analysis & Capital Budgeting": "NPV, IRR, net present value, internal rate of return, capital budgeting, investment appraisal, payback period, cash flows, DCF",
    "Valuation of Financial Assets": "bond valuation, stock valuation, dividend discount model, coupon, yield to maturity, common stock, equity",
    "Portfolio Selection & Diversification": "portfolio theory, diversification, correlation, covariance, efficient frontier, Markowitz, risk reduction",
    "Capital Asset Pricing Model (CAPM)": "CAPM, systematic risk, beta, security market line, market portfolio, risk premium, expected return",
    "Capital Structure & Real Investment Evaluation": "capital structure, leverage, WACC, weighted average cost of capital, Modigliani Miller, debt equity, APV, levered firm",
    "Dividend Policy": "dividend policy, payout ratio, dividend irrelevance, signalling, share buybacks, retained earnings, MM theorem",
    "A Review of Statistical Concepts": "density functions, distribution functions, random variables, mean, variance, covariance, correlation, normal distribution, t-distribution, standard normal, statistical inference, central limit theorem",
    "Bivariate Linear Regression Analysis": "OLS, ordinary least squares, simple regression, dependent variable, regressor, residuals, beta coefficients, MPC, Keynesian consumption, Y = B0 + B1X, estimator properties, unbiasedness, efficiency, consistency, hypothesis testing, t-test",
    "Multiple Linear Regression Analysis": "multiple regression, multiple regressors, F-test, R-squared, adjusted R-squared, multicollinearity, partial effects, control variables",
    "Violations of Some Assumptions": "heteroskedasticity, autocorrelation, serial correlation, non-constant variance, Breusch-Pagan, Durbin-Watson, GLS, White test, robust standard errors, OLS assumptions",
    "Dummy Variables": "dummy variables, binary variables, qualitative data, intercept shift, slope dummy, interaction terms, indicator variable, 0/1 variable, categorical data",
    "Parameter Stability & Structural Change": "Chow test, structural break, parameter stability, regime change, recursive residuals, CUSUM, subsample",
    "Model Selection & Misspecification": "AIC, BIC, RESET test, omitted variable bias, model selection, specification error, overfitting, information criterion, misspecification",
    "Measurement Error": "measurement error, errors in variables, attenuation bias, classical measurement error, reliability ratio, mismeasurement, proxy variable",
    "Endogeneity & Instrumental Variables Estimation": "endogeneity, instrumental variables, IV estimation, 2SLS, two stage least squares, instrument validity, relevance condition, exclusion restriction, simultaneity, reverse causality",
    "Higher Education": "tuition fees, student loans, human capital, graduate premium, returns to education, university funding, access, widening participation",
    "The Economics of Crime": "crime rates, deterrence, rational crime, prison, police, sentencing, social cost of crime, Becker",
    "Population Ageing & Pensions": "ageing population, pension systems, state pension, defined benefit, defined contribution, demographic change, dependency ratio, retirement",
    "Housing and Housing Affordability": "house prices, housing supply, planning constraints, affordability ratio, Help to Buy, mortgage, rental market, homelessness",
    "The Global Financial Crisis & The Great Recession": "financial crisis 2008, subprime mortgage, Lehman Brothers, bank bailouts, recession, credit crunch, systemic risk, contagion",
    "Monetary Policy": "Bank of England, interest rates, inflation targeting, quantitative easing, MPC, base rate, monetary transmission, forward guidance",
    "Fiscal Policy": "government spending, taxation, budget deficit, national debt, fiscal multiplier, austerity, automatic stabilisers, Keynesian fiscal policy",
    "Beyond GDP & The Economics of Happiness": "GDP limitations, wellbeing, happiness economics, Easterlin paradox, human development index, inequality, life satisfaction",
}

def build_topic_context(topics_input):
    """
    Accept either a newline-delimited string or a list of topic name strings.
    Returns (enriched_string, topic_count).
    """
    if isinstance(topics_input, list):
        lines = [t.strip() for t in topics_input if str(t).strip()]
        indexed = [f"{i}: {name}" for i, name in enumerate(lines)]
    else:
        indexed = [l for l in str(topics_input).strip().split("\n") if l.strip()]
        lines = indexed

    enriched = []
    for line in indexed:
        parts = line.split(":", 1)
        idx = parts[0].strip()
        name = parts[1].strip() if len(parts) == 2 else parts[0].strip()
        extra = ""
        for key, ctx in TOPIC_CONTEXT.items():
            if key.lower() in name.lower() or name.lower() in key.lower():
                extra = f" [{ctx}]"
                break
        enriched.append(f"{idx}: {name}{extra}")

    return "\n".join(enriched), len(lines)


@app.route("/parse-paper", methods=["POST"])
def parse_paper():
    """Extract and structure questions from a past exam paper PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = file.read()

    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    filename = file.filename.lower()
    if filename.endswith('.docx'):
        text = extract_docx_text(file_bytes)
    else:
        text = extract_pdf_text(file_bytes)

    if not text or len(text.strip()) < 50:
        return jsonify({"error": "Could not extract text from this PDF"}), 422

    prompt = (
        "You are extracting exam questions from a university past paper.\n\n"
        "Here is the exam paper text:\n\n"
        f"{text[:6000]}\n\n"
        "Extract all questions and return ONLY a valid JSON array, no markdown:\n"
        '[{"number":"1","question":"full question text","marks":10,"topic_hint":"likely topic name"}]\n\n'
        "Rules:\n"
        "- Include every question and sub-question\n"
        "- For sub-questions use number like 1a, 1b\n"
        "- marks should be an integer, use 0 if not shown\n"
        "- topic_hint should be a short 2-5 word description of what topic this tests\n"
        "- Preserve the exact question wording\n"
        "- Return only the JSON array, nothing else"
    )

    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = _message_text(message).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        questions = json.loads(raw)
        return jsonify({"questions": questions, "total": len(questions)})
    except json.JSONDecodeError as e:
        return jsonify({"error": "Could not parse questions: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mark-answer", methods=["POST"])
def mark_answer():
    """Mark a student answer against an exam question using Claude."""
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Missing question or answer"}), 400

    question = data["question"]
    answer = data["answer"]
    marks = data.get("marks", 0)
    notes = data.get("notes", "")

    system = "You are a university economics and finance examiner. Mark student answers fairly and provide constructive feedback."
    if notes:
        system += f"\n\nRelevant lecture notes for context:\n{notes[:2000]}"

    prompt = (
        f"Question ({marks} marks): {question}\n\n"
        f"Student answer: {answer}\n\n"
        "Mark this answer and return ONLY valid JSON, no markdown:\n"
        '{"marks_awarded": 7, "out_of": 10, "percentage": 70, "grade": "Good", '
        '"feedback": "Clear explanation of what was good and what was missing", '
        '"key_points_missed": ["point 1", "point 2"], '
        '"model_answer_hints": "Brief outline of what a full answer would include"}'
    )

    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = _message_text(message).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        result = json.loads(raw)
        return jsonify(result)
    except json.JSONDecodeError as e:
        return jsonify({"error": "Could not parse marking result: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/extract-topics", methods=["POST"])
def extract_topics():
    """Extract structured topic list from a module outline."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get("text", "")
    module_name = data.get("module", "this module")

    prompt = (
        f"You are reading a university module outline for '{module_name}'.\n\n"
        f"Module outline:\n{text[:6000]}\n\n"
        "Extract the list of topics/weeks covered in this module.\n"
        "Return ONLY a valid JSON array, no markdown:\n"
        '[{"id":"topic_1","name":"Full Topic Name","tag":"Week 1"},{"id":"topic_2","name":"Full Topic Name","tag":"Week 2"}]\n\n'
        "Rules:\n"
        "- Use the exact topic names from the outline\n"
        "- tag should be the week number or section (e.g. Week 1, Topic 3, Wks 1-5)\n"
        "- id should be topic_1, topic_2 etc\n"
        "- Include every distinct topic, not just weeks\n"
        "- Return only the JSON array, nothing else"
    )

    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = _message_text(message).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        topics = json.loads(raw)
        return jsonify({"topics": topics})
    except json.JSONDecodeError as e:
        return jsonify({"error": "Could not parse topics: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sort", methods=["POST"])
def sort_topics():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"indices": "none", "debug_topics": None, "debug_text_preview": None, "debug_raw_model_output": None, "debug_topic_count": 0}), 400

    module = data.get("module", "")
    topics = data.get("topics", "")
    text = data.get("text", "").strip()

    if not text or not topics:
        return jsonify({"indices": "none", "debug_topics": None, "debug_text_preview": text[:500], "debug_raw_model_output": None, "debug_topic_count": 0}), 400

    enriched_topics, topic_count = build_topic_context(topics)
    max_valid = topic_count - 1
    snippet = text[:3500]

    system = (
        "You are an academic topic classifier. "
        "Your job is to find the best-matching topics for a piece of lecture content. "
        "Match by concepts, terminology, formulas, and examples — not just exact wording. "
        "Be generous: if there is any reasonable overlap, include the topic. "
        "Only return none if the content is completely unrelated to every topic. "
        "Reply with only digits and commas, or the word none. No other text."
    )

    prompt = (
        f"Module: {module}\n\n"
        f"Topics (with keywords indicating relevance):\n{enriched_topics}\n\n"
        f"Lecture content:\n{snippet}\n\n"
        "Task: identify the 1 to 3 topic numbers this content most strongly covers.\n"
        "- Prefer returning at least 1 match if any topic has even partial relevance\n"
        "- Use the keyword hints in brackets to guide matching\n"
        "- Reply with comma-separated topic numbers only. Example: 0,2\n"
        "- Only reply with none if there is truly zero relevance to any topic"
    )

    debug = {
        "debug_topics": enriched_topics,
        "debug_text_preview": text[:500],
        "debug_raw_model_output": None,
        "debug_topic_count": topic_count,
    }

    try:
        message = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = _message_text(message).strip().lower()
        debug["debug_raw_model_output"] = raw

        if not raw or raw == "none":
            result = "none"
        else:
            nums = re.findall(r'\d+', raw)
            valid = []
            for n in nums:
                if 0 <= int(n) <= max_valid and n not in valid:
                    valid.append(n)
            result = ",".join(valid[:3]) if valid else "none"

        return jsonify({"indices": result, **debug})

    except Exception as e:
        debug["debug_raw_model_output"] = f"EXCEPTION: {str(e)}"
        return jsonify({"indices": "none", "error": str(e), **debug}), 500
def _message_text(message):
    parts = []
    for block in getattr(message, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts).strip()


# ── PDF text extraction ──────────────────────────────────────────────────────
def extract_pdf_text(file_bytes):
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages[:40]:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception:
        return ""


def extract_docx_text(file_bytes):
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
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
        messages=[{"role": "user", "content": prompt}],
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
            messages=[{"role": "user", "content": "Reply with exactly: hello"}],
        )
        return jsonify(
            {
                "ok": True,
                "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
                "response": _message_text(message),
            }
        )
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
                "error": str(e),
                "type": type(e).__name__,
                "trace": traceback.format_exc(),
            }
        ), 500


@app.route("/extract", methods=["POST"])
def extract():
    """Extract text from an uploaded PDF."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = file.read()

    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    filename = file.filename.lower()
    if filename.endswith('.docx'):
        text = extract_docx_text(file_bytes)
    else:
        text = extract_pdf_text(file_bytes)

    if not text or len(text.strip()) < 50:
        return jsonify(
            {"error": "Could not extract text from this PDF. Try saving as .txt instead."}
        ), 422

    return jsonify(
        {
            "text": text,
            "words": len(text.split()),
            "pages": len(text.split("\n")),
        }
    )


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
        return jsonify(
            {
                "error": "Claude returned invalid JSON: " + str(e),
                "type": type(e).__name__,
            }
        ), 500
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "type": type(e).__name__,
                "has_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
                "trace": traceback.format_exc(),
            }
        ), 500


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
            messages=[{"role": "user", "content": prompt}],
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
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM progress WHERE key = %s", (key,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row:
        return jsonify(json.loads(row[0]))
    return jsonify({})


@app.route("/progress", methods=["POST"])
def save_progress():
    """Save progress for a user."""
    key = request.args.get("key", "default")
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO progress (key, data) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data",
                (key, json.dumps(data)),
            )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"saved": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
