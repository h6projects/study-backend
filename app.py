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

import google.generativeai as genai
if os.getenv('GOOGLE_API_KEY'):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

app = Flask(__name__)
CORS(app, origins="*")
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=30.0)
claude_vision = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=90.0)

ROUTE_PROVIDERS = {
    'summarise':      'gemini',
    'process_notes':  'gemini',
    'sort':           'gemini',
    'extract_topics': 'gemini',
    'lesson':         'claude',
    'quiz':           'claude',
    'flashcards':     'claude',
    'fill_blanks':    'claude',
    'mark_answer':    'claude',
    'parse_paper':    'claude',
}

def _get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
    conn.commit()
    return conn

TOPIC_CONTEXT = {
    "Overview of the Financial System & Interest Rates": "L1 L2 overview financial system flow of funds financial intermediaries channelling funds liquidity price discovery meaning of interest rates simple interest compound interest present value yield to maturity bond prices nominal real interest rates Fisher equation",
    "Measures of Risk & Behaviour of Interest Rates": "L3 L4 measures of risk standard deviation variance expected return risk premium portfolio risk behaviour of interest rates loanable funds theory liquidity preference Keynes supply demand bonds interest rate determination",
    "Term Structure of Interest Rates & Stock Market": "L5 L6 term structure yield curve expectations theory liquidity premium theory segmented markets theory stock market equity valuation dividend growth model market microstructure bid ask spread dealers brokers",
    "Efficient Market Hypothesis & Derivative Markets": "L7 L8 efficient market hypothesis EMH weak form semi-strong strong form random walk anomalies derivatives options futures forwards hedging speculation arbitrage Black Scholes",
    "Behavioural Finance": "L9 behavioural finance prospect theory loss aversion overconfidence herding anchoring mental accounting market anomalies investor psychology irrational behaviour",
    "Introduction to Foreign Exchange Markets": "L10 foreign exchange FX market spot rate forward rate exchange rate currency appreciation depreciation intervention",
    "Exchange Rate Changes & Interest Rate Parity": "L11 L12 exchange rate changes interest rate parity covered uncovered IRP purchasing power parity arbitrage carry trade",
    "Purchasing Power Parity & International Financial Systems": "L13 L14 purchasing power parity PPP absolute relative law of one price international financial system Bretton Woods IMF World Bank balance of payments",
    "Financial Institutions & Non-Financial Institutions": "L15 L16 commercial banks investment banks mutual funds insurance companies pension funds non-bank financial intermediaries shadow banking",
    "Risk Management and Financial Institutions": "L17 risk management financial institutions credit risk market risk operational risk Basel accords capital requirements VaR value at risk",
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
        f"{text[:50000]}\n\n"
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
        raw = ai_generate(prompt, max_tokens=3000, route='parse_paper').strip()
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
        system += f"\n\nRelevant lecture notes for context:\n{notes[:10000]}"

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
        raw = ai_generate(prompt, system=system, max_tokens=800, route='mark_answer').strip()
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
        f"Module outline:\n{text[:20000]}\n\n"
        "Extract the list of topics/weeks covered in this module.\n"
        "Return ONLY a valid JSON array, no markdown:\n"
        '[{"id":"topic_1","name":"Full Topic Name","tag":"Week 1","lectureHint":"L1-L2"},{"id":"topic_2","name":"Full Topic Name","tag":"Week 2","lectureHint":"L3"}]\n\n'
        "Rules:\n"
        "- Use the exact topic names from the outline\n"
        "- tag should be the week number or section (e.g. Week 1, Topic 3, Wks 1-5)\n"
        "- id should be topic_1, topic_2 etc\n"
        "- Include every distinct topic, not just weeks\n"
        "- lectureHint should be the specific lecture or week reference for this topic extracted from the outline (e.g. 'L1-L2', 'Wk 7', 'Topic 3-4') — keep it short\n"
        "- Return only the JSON array, nothing else"
    )

    try:
        raw = ai_generate(prompt, max_tokens=1500, route='extract_topics').strip()
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
    snippet = text[:20000]

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
        raw = ai_generate(prompt, system=system, max_tokens=20, route='sort').strip().lower()
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


# ── AI provider abstraction ───────────────────────────────────────────────────
def ai_generate(prompt, system=None, max_tokens=1400, model=None, route=None):
    """Primary generation function — routes to active provider.

    Provider resolution order:
    1. ROUTE_PROVIDERS[route] if route is given
    2. AI_PROVIDER env var
    3. 'claude' default
    Falls back to Claude if Gemini is selected but GOOGLE_API_KEY is not set.
    """
    if route and route in ROUTE_PROVIDERS:
        provider = ROUTE_PROVIDERS[route]
    else:
        provider = os.getenv('AI_PROVIDER', 'claude')
    if provider == 'gemini' and not os.getenv('GOOGLE_API_KEY'):
        provider = 'claude'
    if provider == 'claude':
        return _claude_generate(prompt, system, max_tokens, model)
    elif provider == 'gemini':
        try:
            return _gemini_generate(prompt, system, max_tokens, model)
        except Exception as e:
            err = str(e).lower()
            if 'quota' in err or '429' in err or 'exhausted' in err or 'billing' in err:
                print(f'[ai_generate] Gemini quota/billing error on route={route}, falling back to Claude: {type(e).__name__}')
                return _claude_generate(prompt, system, max_tokens, model)
            raise
    else:
        raise ValueError(f'Unknown AI provider: {provider}')

def _claude_generate(prompt, system=None, max_tokens=1400, model=None):
    """Claude generation via Anthropic SDK."""
    model = model or 'claude-sonnet-4-20250514'
    msg_params = {
        'model': model,
        'max_tokens': max_tokens,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    if system:
        msg_params['system'] = system
    message = claude.messages.create(**msg_params)
    return _message_text(message)

def _gemini_generate(prompt, system=None, max_tokens=1400, model=None):
    """Gemini generation via google-generativeai SDK."""
    model_name = model or 'gemini-1.5-flash'
    gemini_model = genai.GenerativeModel(model_name)
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = gemini_model.generate_content(full_prompt)
    return response.text


# ── PDF text extraction ──────────────────────────────────────────────────────
def ai_vision(image_data, prompt):
    """Vision analysis — currently Claude-specific, wrappable for future providers."""
    msg = claude_vision.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
    )
    return _message_text(msg)


def _describe_page_visuals(img_b64, page_num):
    """Send a rendered PDF page to vision AI and get a diagram description."""
    try:
        prompt = (
            f"This is page {page_num} of university lecture notes. "
            "Describe any diagrams, charts, curves, graphs or tables visible. "
            "Use exact axis labels, variable names and economic notation. "
            "If it is an economic diagram name the model (e.g. IS-LM, AD-AS, Phillips curve). "
            "If it is a regression output table, transcribe the key coefficient values and significance levels. "
            "If the page contains only text with no diagrams, respond with exactly: TEXT_ONLY"
        )
        desc = ai_vision(img_b64, prompt).strip()
        if desc == "TEXT_ONLY":
            return None
        return desc
    except Exception:
        return None


def extract_pdf_text(file_bytes):
    """Extract text from PDF, with Claude Vision descriptions for pages containing images."""
    try:
        import fitz  # pymupdf
        import base64
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        vision_count = 0
        max_vision_pages = 5

        for page_num in range(min(len(doc), 40)):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text + "\n"

            # Only send to Vision if the page has embedded raster images
            # (charts exported from Excel/STATA, screenshots, diagrams saved as images)
            has_images = len(page.get_images()) > 0
            # Also catch pages with many vector drawings (IS-LM curves drawn in PowerPoint)
            has_complex_drawings = len(page.get_drawings()) > 8

            if (has_images or has_complex_drawings) and vision_count < max_vision_pages:
                try:
                    mat = fitz.Matrix(1.5, 1.5)  # ~108 DPI — fast but readable
                    pix = page.get_pixmap(matrix=mat)
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                    desc = _describe_page_visuals(img_b64, page_num + 1)
                    if desc:
                        text += f"\n--- Diagram on page {page_num + 1} ---\n{desc}\n\n"
                        vision_count += 1
                except Exception:
                    pass  # skip failed vision pages silently

        doc.close()
        return text.strip()

    except Exception:
        # Fallback to pypdf if pymupdf is unavailable or fails
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
    system = (
        "You are a university tutor helping a 2nd year economics and finance student at UoB prepare for exams. "
        "Write with clarity and precision — explain concepts directly and clearly, not with unnecessarily complex language. "
        "Every sentence should help the student understand and remember the concept. "
        "Use the lecturer's exact notation and formulas. "
        "Be rigorous but never obscure. "
        "Avoid filler phrases like 'it is worth noting', 'it is important to recognise', 'the discipline requires'. "
        "Get straight to the point."
    )
    if module_outline:
        system += f"\n\nModule outline for context:\n{module_outline}"

    prompt = (
        f"Create a structured lesson on '{topic_name}' using these lecture notes:\n\n"
        f"{text[:50000]}\n\n"
        "Return ONLY a valid JSON object, no markdown, no backticks:\n"
        '{"title":"...","key_concepts":["concept 1","concept 2","concept 3"],'
        '"slides":[{"title":"slide title","body":"3-5 clear sentences that explain the concept directly. Start with what it is, then explain how it works, then give the key insight or condition. Use plain academic English — precise but readable.","highlight":"exact formula with all variables defined, or precise theorem statement"}],'
        '"exam_tips":["tip 1","tip 2"]}'
        "\n\nInclude exactly 6 slides. Use the lecturer's exact notation throughout. "
        "Each slide body: 3-5 sentences, direct and clear. Start with what the concept is. Explain the mechanism. State the key condition or implication. "
        "Each highlight: exact formula with every variable defined, or a precise theorem/condition statement. "
        "Do not use phrases like 'in simple terms', 'basically', 'it is worth noting', or 'importantly'. "
        "Do not pad sentences. Every word should earn its place."
    )

    raw = ai_generate(prompt, system=system, max_tokens=2500, route='lesson')
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


@app.route("/debug_api")
def debug_api():
    google_key_set = bool(os.getenv('GOOGLE_API_KEY'))
    effective = {}
    for route, provider in ROUTE_PROVIDERS.items():
        if provider == 'gemini' and not google_key_set:
            effective[route] = 'claude (fallback — no GOOGLE_API_KEY)'
        else:
            effective[route] = provider
    return jsonify({
        "anthropic_key": bool(os.getenv('ANTHROPIC_API_KEY')),
        "google_key": google_key_set,
        "ai_provider_env": os.getenv('AI_PROVIDER', 'claude'),
        "route_providers": ROUTE_PROVIDERS,
        "effective_providers": effective,
    })


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


@app.route("/process-notes", methods=["POST"])
def process_notes():
    """Extract structured academic content from lecture notes."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    topic_name = data.get("topic", "this topic")

    if len(text.strip()) < 100:
        return jsonify({"error": "Notes too short to process"}), 400

    system = (
        "You are extracting the most important academic content from university lecture notes "
        "for a 2nd year economics and finance student at the University of Birmingham. "
        "Extract only what is genuinely important — key concepts with precise definitions, "
        "critical formulas with all variables explained, exam-relevant points, and common mistakes to avoid. "
        "Be selective and precise. Nothing generic. Use the lecturer's exact notation."
    )

    prompt = (
        f"Extract the most important content from these lecture notes on '{topic_name}':\n\n"
        f"{text[:50000]}\n\n"
        "Return ONLY a valid JSON object, no markdown:\n"
        '{"key_concepts":[{"term":"...","definition":"one precise sentence","formula":"optional — exact notation or empty string"}],'
        '"formulas":[{"name":"...","formula":"exact notation","variables":"what each symbol means"}],'
        '"exam_tips":["precise point 1","precise point 2"],'
        '"common_mistakes":["mistake 1","mistake 2"]}\n\n'
        "Rules:\n"
        "- key_concepts: 4-8 most important concepts. Definition must be precise, not vague.\n"
        "- formulas: only actual mathematical formulas. Include all variables.\n"
        "- exam_tips: 3-6 points. Specific to this topic — not generic advice.\n"
        "- common_mistakes: 2-4 specific errors students make on this topic.\n"
        "- If a section has nothing relevant, return an empty array."
    )

    try:
        raw = ai_generate(prompt, system=system, max_tokens=3000, route='process_notes')
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        result = json.loads(raw)
        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "type": type(e).__name__, "traceback": tb}), 500


@app.route("/quiz", methods=["POST"])
def quiz():
    """Generate quiz questions from lesson content using Claude."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    slides = data.get("slides", "")
    topic_name = data.get("topic", "this topic")
    mode = data.get("mode", "learn")
    outline = data.get("outline", "")
    mark_scheme = data.get("markScheme", "")
    topics = data.get("topics", [])

    content = f"Lecture notes:\n{text[:50000]}"
    if slides:
        content += f"\n\nLesson slides summary:\n{slides}"
    if mark_scheme:
        content += f"\n\nMark scheme / model answer guidance:\n{mark_scheme[:20000]}"

    topics_instruction = ""
    if topics:
        topics_instruction = f"Spread questions across these module topics: {', '.join(topics[:20])}.\n"

    if mode == "speed":
        prompt = (
            f"Create 20 rapid-fire multiple choice questions on '{topic_name}'.\n\n"
            f"{topics_instruction}"
            f"Content:\n{content}\n\n"
            "Requirements:\n"
            "- Questions should be short and punchy — test precise recall of definitions, formulas and conditions\n"
            "- Use exact academic terminology and notation from the notes\n"
            "- Options should be brief (1-5 words each where possible)\n"
            "- Keep explanations to one precise sentence\n"
            "- Cover as many distinct concepts as possible across all topics\n\n"
            "Return ONLY a valid JSON array, no markdown:\n"
            '[{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"...","concept":"..."}]'
            "\n\ncorrect is 0-based index. Generate exactly 20 questions."
        )
    elif mode == "exam":
        prompt = (
            f"Create 6 exam-style multiple choice questions on '{topic_name}' as they appear in a University of Birmingham 2nd year economics exam.\n\n"
            f"{topics_instruction}"
            f"Content:\n{content}\n\n"
            "Requirements:\n"
            "- Questions must be genuinely difficult — requiring application and analysis, not just recall\n"
            "- Include multi-step numerical questions requiring calculation where the topic allows\n"
            "- Test understanding of when models break down and what assumptions are violated\n"
            "- Use precise academic notation from the notes\n"
            "- Wrong options must be plausible misconceptions a well-prepared student might choose\n"
            "- explanation: full academic explanation of why the correct answer is right and specifically why each wrong option fails\n\n"
            "Return ONLY a valid JSON array, no markdown:\n"
            '[{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"...","concept":"..."}]'
            "\n\ncorrect is 0-based index. Generate exactly 6 questions."
        )
    else:
        prompt = (
            f"Create 6 multiple choice questions testing genuine academic understanding of '{topic_name}'.\n\n"
            f"Content:\n{content}\n\n"
            "Requirements:\n"
            "- Questions must require real understanding, not just recall — test application of concepts\n"
            "- Include at least 2 multi-step calculation or derivation questions where the topic allows\n"
            "- Test knowledge of model assumptions and when models break down\n"
            "- Use the exact notation from the lecture notes\n"
            "- Wrong options should be common misconceptions or near-misses, not obviously wrong\n"
            "- explanation: a full academic paragraph explaining the economic intuition and why wrong answers fail\n"
            "- Mix: 2 conceptual, 2 application/calculation, 2 critical analysis\n\n"
            "Return ONLY a valid JSON array, no markdown:\n"
            '[{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"...","concept":"..."}]'
            "\n\ncorrect is 0-based index. Generate exactly 6 questions."
        )

    try:
        raw = ai_generate(prompt, max_tokens=4000, route='quiz')
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


@app.route("/flashcards", methods=["POST"])
def flashcards():
    """Generate flashcard pairs from lecture notes."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get("text", "")
    topic_name = data.get("topic", "this topic")

    system = (
        "You are creating flashcards for a 2nd year undergraduate economics and finance student "
        "at the University of Birmingham preparing for exams. "
        "Each card tests one precise academic concept. "
        "Fronts must be precise academic questions or 'State the [theorem/condition/formula]' style prompts. "
        "Backs must be complete formal answers with the exact formula, all assumptions stated, and full economic intuition. "
        "Use the lecturer's exact notation. Never use dumbed-down language or simplified explanations."
    )

    prompt = (
        f"Create 8-12 flashcards for the topic '{topic_name}' using these lecture notes:\n\n"
        f"{text[:50000]}\n\n"
        "Return ONLY a valid JSON array, no markdown:\n"
        '[{"front": "State the OLS estimator for \u03b2\u2081 in simple regression", "back": "\u03b2\u0302\u2081 = \u03a3(x\u1d62 - x\u0304)(y\u1d62 - \u0233) / \u03a3(x\u1d62 - x\u0304)\u00b2. Under Gauss-Markov assumptions (linearity, exogeneity, homoskedasticity, no autocorrelation), \u03b2\u0302\u2081 is BLUE: unbiased (E[\u03b2\u0302\u2081]=\u03b2\u2081), consistent, and efficient among all linear unbiased estimators."}]\n\n'
        "Generate between 8 and 12 cards. "
        "Fronts: precise question or 'State the...' prompt. "
        "Backs: complete formal answer with formula, assumptions, and economic intuition. "
        "No simplified language — 2nd year university standard throughout."
    )

    try:
        raw = ai_generate(prompt, system=system, max_tokens=2500, route='flashcards').strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        cards = json.loads(raw)
        return jsonify({"cards": cards})
    except json.JSONDecodeError as e:
        return jsonify({"error": "Could not parse flashcards: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fill-blanks", methods=["POST"])
def fill_blanks():
    """Generate fill-in-the-blank exercises from lecture notes."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data.get("text", "")
    topic_name = data.get("topic", "this topic")

    prompt = (
        f"Create 5 fill-in-the-blank exercises for the topic '{topic_name}' "
        f"using these lecture notes:\n\n{text[:50000]}\n\n"
        "Focus on completing formal proofs or derivations, filling in formula components, "
        "and stating conditions and assumptions precisely. "
        "Each blank should be a specific term, symbol, formula component, or precise condition — "
        "not a vague summary word.\n\n"
        "Return ONLY a valid JSON array, no markdown:\n"
        '[{"sentence": "Under the Gauss-Markov assumptions, OLS is ___ among all linear unbiased estimators", "answer": "BLUE (Best Linear Unbiased Estimator)", "hint": "The efficiency property — OLS has minimum variance in the class of linear unbiased estimators"}]\n\n'
        "Generate exactly 5 exercises. Use ___ to mark the blank. "
        "Prioritise derivation steps, formal conditions, precise mathematical statements, and exact model assumptions."
    )

    try:
        raw = ai_generate(prompt, max_tokens=1500, route='fill_blanks').strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        exercises = json.loads(raw)
        return jsonify({"exercises": exercises})
    except json.JSONDecodeError as e:
        return jsonify({"error": "Could not parse exercises: " + str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarise", methods=["POST"])
def summarise():
    """Intelligently compress long document text to fit within the notes cap."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    topic = data.get("topic", "this topic")

    # Under threshold — return as-is
    if len(text) <= 40000:
        return jsonify({"text": text, "summarised": False})

    chunk_size = 15000
    overlap = 1000
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_system = (
                "You are extracting examinable academic content from university lecture notes. "
                "Extract every key concept, definition, formula, model, example, and piece of examinable content. "
                "Be comprehensive — nothing that could appear in an exam should be omitted. "
                "Use the lecturer's exact notation."
            )
            chunk_prompt = (
                f"Lecture notes for '{topic}' (part {i+1} of {len(chunks)}):\n\n{chunk}\n\n"
                "Extract all examinable content from this section in structured form."
            )
            chunk_summaries.append(ai_generate(chunk_prompt, system=chunk_system, max_tokens=2000, route='summarise'))
        except Exception as e:
            chunk_summaries.append(chunk[:3000])  # fallback: use raw chunk excerpt

    combined = "\n\n".join(chunk_summaries)

    # If combined fits, return directly
    if len(combined) <= 45000:
        return jsonify({
            "text": combined,
            "summarised": True,
            "original_length": len(text),
            "chunks": len(chunks)
        })

    # Final consolidation pass
    try:
        final_system = (
            "You are consolidating extracted lecture notes into one structured study document. "
            "Preserve all key concepts, definitions, formulas, and examinable content. "
            "Organise clearly with section headings. Use the lecturer's exact notation."
        )
        final_prompt = (
            f"Consolidated notes for '{topic}':\n\n{combined[:80000]}\n\n"
            "Produce one structured document covering all the above content. "
            "Remove redundancy but keep every distinct concept, formula and example."
        )
        final_text = ai_generate(final_prompt, system=final_system, max_tokens=4000, route='summarise')
    except Exception:
        final_text = combined[:45000]

    return jsonify({
        "text": final_text,
        "summarised": True,
        "original_length": len(text),
        "chunks": len(chunks)
    })


@app.route("/clear-custom-topics", methods=["POST"])
def clear_custom_topics():
    """Remove a module's customTopics entry from stored progress."""
    data = request.get_json()
    if not data or "key" not in data or "modId" not in data:
        return jsonify({"error": "Missing key or modId"}), 400
    key = data["key"]
    mod_id = data["modId"]
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM progress WHERE key = %s", (key,))
            row = cur.fetchone()
        if not row:
            return jsonify({"cleared": False, "reason": "no progress found"})
        progress = json.loads(row[0])
        if "customTopics" not in progress or mod_id not in progress["customTopics"]:
            return jsonify({"cleared": False, "reason": "modId not in customTopics"})
        del progress["customTopics"][mod_id]
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE progress SET data = %s WHERE key = %s",
                (json.dumps(progress), key),
            )
        conn.commit()
    finally:
        conn.close()
    return jsonify({"cleared": True, "modId": mod_id})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
