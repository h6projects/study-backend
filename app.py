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

from google import genai as google_genai
google_client = google_genai.Client(api_key=os.getenv('GOOGLE_API_KEY')) if os.getenv('GOOGLE_API_KEY') else None

try:
    from supabase import create_client as _sb_create
    supabase_client = _sb_create(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY')) if os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_SERVICE_KEY') else None
except Exception as _sb_e:
    print(f'[supabase] init failed: {_sb_e}')
    supabase_client = None

app = Flask(__name__)
CORS(app, origins=["*"], allow_headers=["*"], supports_credentials=False)
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
    'vision':         'gemini',
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
            print(f'[ai_generate] Gemini error on route={route}: {type(e).__name__}: {str(e)[:200]}')
            if 'quota' in err or '429' in err or 'exhausted' in err or 'billing' in err or '503' in err or 'unavailable' in err:
                print(f'[ai_generate] Gemini unavailable — falling back to Claude')
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
    """Gemini generation via google-genai SDK."""
    if not google_client:
        raise ValueError('GOOGLE_API_KEY not set')
    model_name = model or 'gemini-2.5-flash'
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    print(f"[Gemini] Using model: {model_name}")
    print(f"[Gemini] Prompt length: {len(full_prompt)}")
    response = google_client.models.generate_content(
        model=model_name,
        contents=full_prompt
    )
    print(f"[Gemini] Response received, length: {len(response.text)}")
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


def extract_page_with_gemini(page_image_bytes, page_num, page_text=''):
    """Use Gemini Vision to extract structured content from a single PDF page.
    Zero Claude/Anthropic API calls — Gemini only."""
    if not google_client:
        return None
    try:
        from google.genai import types as genai_types
        prompt = (
            "You are analysing a university lecture slide for a Money, Banking and Finance student preparing for exams.\n\n"
            "Extract ALL visual content from this slide. Be inclusive — extract anything that is not plain bullet-point text, "
            "including: charts, graphs, diagrams, flowcharts, arrows showing relationships, illustrative figures, grids, "
            "matrices, tables, balance sheets, timelines, and any other non-text visual element.\n\n"
            "Analyse this slide and return JSON:\n"
            "{\n"
            '  "has_table": true/false,\n'
            '  "has_diagram": true/false,\n'
            '  "has_meaningful_content": true/false,\n'
            '  "table_markdown": "if has_table: full markdown table with headers inferred from context. Include ALL rows and columns.",\n'
            '  "diagram_type": "name the visual e.g. Flowchart, Supply and Demand, Liquidity Spiral, Bar Chart, Illustrative Figure, Balance Sheet, Timeline — be descriptive if no standard name applies",\n'
            '  "diagram_description": "Describe the ECONOMIC MEANING: what does this show, what are the axes or entities, what do arrows/curves/cells represent, what is the exam takeaway. Include specific values, labels, and numbers visible in the diagram.",\n'
            '  "diagram_svg_hint": "Describe layout precisely: entities/nodes, arrows and their direction/labels, axis labels, key values, spatial arrangement",\n'
            '  "clean_text": "main academic text only — exclude slide numbers, lecturer name, university name, module name, term name"\n'
            "}\n\n"
            "If a slide contains BOTH a table AND a diagram, set both has_table and has_diagram to true and populate all relevant fields.\n"
            "Return ONLY valid JSON."
        )
        response = google_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[genai_types.Part.from_bytes(data=page_image_bytes, mime_type='image/png'), prompt]
        )
        raw = response.text.strip()
        if raw.startswith('```json'):
            raw = raw[7:]
        elif raw.startswith('```'):
            raw = raw[3:]
        if raw.endswith('```'):
            raw = raw[:-3]
        result = json.loads(raw.strip())
        print(f'[extract_page_with_gemini] Page {page_num}: meaningful={result.get("has_meaningful_content")}, table={result.get("has_table")}, diagram={result.get("has_diagram")}')
        return result
    except Exception as e:
        print(f'[extract_page_with_gemini] Page {page_num} failed: {type(e).__name__}: {str(e)[:120]}')
        return None


def _upload_to_storage(img_bytes, filename, page_num):
    """Upload page PNG to Supabase Storage bucket 'diagrams'. Returns public URL or empty string."""
    if not supabase_client:
        print(f'[storage] supabase_client is None — skipping upload for {filename} page {page_num}')
        return ''
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', filename.replace('.pdf', '').replace('.PDF', ''))
    path = f"{safe_name}_{page_num}.png"
    print(f'[storage] Uploading {path} to Supabase...')
    try:
        upload_result = supabase_client.storage.from_('diagrams').upload(
            path=path,
            file=img_bytes,
            file_options={"content-type": "image/png", "upsert": "true"}
        )
        print(f'[storage] Upload result: {upload_result}')
    except Exception as e:
        print(f'[storage] Upload FAILED for {path}: {type(e).__name__}: {e}')
        return ''
    try:
        url = supabase_client.storage.from_('diagrams').get_public_url(path)
        print(f'[storage] Public URL: {url}')
        return url if isinstance(url, str) else ''
    except Exception as e:
        print(f'[storage] get_public_url FAILED for {path}: {type(e).__name__}: {e}')
        return ''


def extract_pdf_full(file_bytes, filename='document'):
    """Unified extraction pipeline — pymupdf text + Gemini Vision + Supabase Storage.
    Zero Claude/Anthropic API calls during extraction."""
    diagrams = []
    tables = []
    raw_text = ""
    page_segments = []

    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        # Step 1: Extract raw text via pymupdf (no AI)
        for page_num in range(min(len(doc), 40)):
            raw_text += doc[page_num].get_text() + "\n"

        # Step 2: Gemini Vision — sequential, one page at a time
        total_pages = min(len(doc), 40)
        vision_limit = min(len(doc), 20)
        vision_done = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            page_raw = page.get_text()

            if not google_client or page_num >= vision_limit:
                page_segments.append(f"<<PAGE:{page_num+1}>>\n{page_raw}")
                continue

            try:
                mat = fitz.Matrix(2.083, 2.083)  # 150 DPI
                img_bytes = page.get_pixmap(matrix=mat).tobytes("png")
                result = extract_page_with_gemini(img_bytes, page_num + 1, page_raw)
                if not result:
                    page_segments.append(f"<<PAGE:{page_num+1}>>\n{page_raw}")
                    continue
                if not result.get('has_meaningful_content', True):
                    continue
                clean = (result.get('clean_text') or page_raw).strip()
                if clean:
                    page_segments.append(f"<<PAGE:{page_num+1}>>\n{clean}")
                if result.get('has_table') and result.get('table_markdown'):
                    table_md = result['table_markdown'].strip()
                    page_segments.append(table_md)
                    img_url = _upload_to_storage(img_bytes, filename, page_num + 1)
                    tables.append({"page": page_num + 1, "markdown": table_md, "image_url": img_url})
                if result.get('has_diagram'):
                    dtype = result.get('diagram_type') or 'Diagram'
                    ddesc = (result.get('diagram_description') or '').strip()
                    dsvg = (result.get('diagram_svg_hint') or '').strip()
                    img_url = _upload_to_storage(img_bytes, filename, page_num + 1)
                    diagrams.append({"page": page_num + 1, "type": dtype, "description": ddesc,
                                      "image_url": img_url, "svg_hint": dsvg})
                    block = f"\n--- DIAGRAM: {dtype} ---\n"
                    if ddesc: block += ddesc + "\n"
                    if dsvg: block += f"SVG_HINT: {dsvg}\n"
                    block += f"PAGE: {page_num + 1}\n--- END DIAGRAM ---\n"
                    page_segments.append(block)
                vision_done += 1
            except Exception as e:
                print(f'[extract_pdf_full] page {page_num+1} failed: {e}')
                page_segments.append(f"<<PAGE:{page_num+1}>>\n{page_raw}")

        print(f'[extract_pdf_full] {filename}: vision={vision_done}/{vision_limit} pages, diagrams={len(diagrams)}, tables={len(tables)}')

        doc.close()
        merged = '\n\n'.join(page_segments) if page_segments else raw_text
        text = clean_extracted_text(merged.strip())

    except Exception as e:
        print(f'[extract_pdf_full] fitz failed: {e}')
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages[:40]:
                raw_text += (page.extract_text() or "") + "\n"
            text = clean_extracted_text(raw_text.strip())
        except Exception:
            text = ''

    return {
        "text": text.strip(),
        "raw_text": raw_text.strip(),
        "diagrams": diagrams,
        "tables": tables,
        "has_enhanced": bool(diagrams or tables)
    }


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


@app.route("/debug_gemini")
def debug_gemini():
    if not google_client:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 400
    try:
        models = [m.name for m in google_client.models.list()]
        return jsonify({"models": models, "count": len(models)})
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/admin/clean-notes", methods=["POST"])
def admin_clean_notes():
    """One-time migration: run clean_extracted_text on all stored notes."""
    admin_key = os.getenv("ADMIN_KEY")
    if not admin_key:
        return jsonify({"error": "ADMIN_KEY not set on server"}), 403
    if request.headers.get("X-Admin-Key") != admin_key:
        return jsonify({"error": "Forbidden"}), 403

    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, data FROM progress")
            rows = cur.fetchall()
    finally:
        conn.close()

    summary = {"rows_processed": 0, "topics_cleaned": [], "topics_skipped": []}

    for db_key, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except Exception:
            continue

        notes = data.get("notes", {})
        if not notes:
            continue

        raw_notes = data.get("rawNotes", {})
        changed = False

        for topic_key, text in list(notes.items()):
            if not isinstance(text, str) or len(text.strip()) < 50:
                continue
            # Skip if we already have a raw backup for this topic
            if topic_key in raw_notes:
                summary["topics_skipped"].append(topic_key)
                continue

            cleaned = clean_extracted_text(text)
            lines_before = len([l for l in text.split('\n') if l.strip()])
            lines_after = len([l for l in cleaned.split('\n') if l.strip()])
            removed = lines_before - lines_after

            raw_notes[topic_key] = text
            notes[topic_key] = cleaned
            changed = True
            summary["topics_cleaned"].append({
                "topic": topic_key,
                "lines_before": lines_before,
                "lines_after": lines_after,
                "lines_removed": removed,
            })

        if changed:
            data["notes"] = notes
            data["rawNotes"] = raw_notes
            conn2 = _get_db()
            try:
                with conn2.cursor() as cur:
                    cur.execute(
                        "UPDATE progress SET data = %s WHERE key = %s",
                        (json.dumps(data), db_key),
                    )
                conn2.commit()
            finally:
                conn2.close()
            summary["rows_processed"] += 1

    return jsonify(summary)


def _insert_page_markers_heuristic(text):
    """Insert <<PAGE:N>> markers using text heuristics — no AI call, instant.

    A new page starts when:
    - Line is ALL CAPS with 4+ chars (slide title)
    - Line matches week/lecture/topic/section/chapter + number
    - Line ends with ':' and is short (<70 chars) — section heading
    - A blank line is followed by a numbered list restart (1. or 1))
    - Every MAX_LINES lines as a hard fallback
    """
    import re
    MAX_LINES = 35  # hard fallback: new page every 35 lines

    lines = text.split('\n')
    out = []
    page = 0
    lines_since_break = 0

    def is_page_break(line, prev_blank):
        t = line.strip()
        if not t:
            return False
        # ALL CAPS heading (4+ non-space chars)
        if re.match(r'^[A-Z][A-Z\s\d&/:()_\-]{3,}$', t) and len(t) >= 4:
            return True
        # Week/Lecture/Topic/Section/Chapter heading
        if re.match(r'^(week|lecture|topic|section|chapter|slide)\s*\d', t, re.I):
            return True
        # Short heading ending with colon
        if t.endswith(':') and len(t) < 70 and prev_blank:
            return True
        # Numbered list restart after blank line (1. or 1))
        if prev_blank and re.match(r'^1[.)]\s', t):
            return True
        return False

    prev_blank = True  # treat start as after a blank line
    for i, line in enumerate(lines):
        stripped = line.strip()
        force_break = lines_since_break >= MAX_LINES and stripped

        if force_break or is_page_break(line, prev_blank):
            page += 1
            out.append(f'<<PAGE:{page}>>')
            lines_since_break = 0

        out.append(line)
        if stripped:
            lines_since_break += 1
        prev_blank = not stripped

    # If no breaks were inserted at all, wrap everything as page 1
    if page == 0:
        return f'<<PAGE:1>>\n' + text

    return '\n'.join(out)


@app.route("/admin/add-page-markers", methods=["POST"])
def admin_add_page_markers():
    """One-time migration: insert <<PAGE:N>> markers into stored notes that lack them.

    Uses fast Python heuristics (no AI call) to detect slide boundaries so the
    route completes well within Railway's CDN timeout.
    """
    import traceback
    admin_key = os.getenv("ADMIN_KEY", "studyai-admin")
    if request.headers.get("X-Admin-Key") != admin_key:
        return jsonify({"error": "Forbidden"}), 403

    # dry_run=1: count topics needing markers without writing anything
    dry_run = request.args.get('dry_run') == '1'

    try:
        conn = _get_db()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT key, data FROM progress")
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": "db_fetch_failed", "detail": traceback.format_exc()[-500:]}), 500

    summary = {"rows_processed": 0, "topics_updated": [], "topics_skipped": [], "errors": []}

    for db_key, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except Exception:
            continue

        notes = data.get("notes", {})
        if not notes:
            continue

        changed = False
        for topic_key, text in list(notes.items()):
            if not isinstance(text, str) or len(text.strip()) < 100:
                summary["topics_skipped"].append(topic_key)
                continue
            if "<<PAGE:" in text:
                summary["topics_skipped"].append(topic_key)
                continue

            if dry_run:
                summary["topics_updated"].append(topic_key)
                continue

            try:
                marked = _insert_page_markers_heuristic(text)
                notes[topic_key] = marked
                changed = True
                summary["topics_updated"].append(topic_key)
            except Exception as e:
                summary["errors"].append(f"{topic_key}: {str(e)[:80]}")

        if changed:
            data["notes"] = notes
            conn2 = _get_db()
            try:
                with conn2.cursor() as cur:
                    cur.execute(
                        "UPDATE progress SET data = %s WHERE key = %s",
                        (json.dumps(data), db_key),
                    )
                conn2.commit()
            finally:
                conn2.close()
            summary["rows_processed"] += 1

    return jsonify(summary)


@app.route("/admin/fix-topic-markers/<topic_id>", methods=["POST"])
def admin_fix_topic_markers(topic_id):
    """Strip and re-insert <<PAGE:N>> markers for a single topic using the heuristic.

    Notes live in progress.data (JSON blob) under data['notes'][topic_id].
    Returns marker count and a preview of where each marker was placed.
    """
    import re, traceback
    admin_key = os.getenv("ADMIN_KEY", "studyai-admin")
    if request.headers.get("X-Admin-Key") != admin_key:
        return jsonify({"error": "Forbidden"}), 403

    try:
        conn = _get_db()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT key, data FROM progress")
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": "db_fetch_failed", "detail": traceback.format_exc()[-500:]}), 500

    found_row_key = None
    found_text = None
    found_data = None

    for db_key, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except Exception:
            continue
        notes = data.get("notes", {})
        if topic_id in notes and isinstance(notes[topic_id], str):
            found_row_key = db_key
            found_text = notes[topic_id]
            found_data = data
            break

    if found_row_key is None:
        return jsonify({"error": f"topic '{topic_id}' not found in any progress row"}), 404

    original_len = len(found_text)
    original_markers = len(re.findall(r'<<PAGE:\d+>>', found_text))

    # Strip all existing markers
    stripped = re.sub(r'<<PAGE:\d+>>\n?', '', found_text).strip()

    # Re-run heuristic on clean text
    marked = _insert_page_markers_heuristic(stripped)

    # Build preview: first 80 chars of text after each marker
    previews = []
    for m in re.finditer(r'<<PAGE:(\d+)>>(.*?)(?=<<PAGE:\d+>>|$)', marked, re.DOTALL):
        page_n = m.group(1)
        snippet = m.group(2).strip()[:80].replace('\n', ' ')
        previews.append({"page": int(page_n), "preview": snippet})

    new_markers = len(previews)

    # Save back
    found_data["notes"][topic_id] = marked
    try:
        conn3 = _get_db()
        try:
            with conn3.cursor() as cur:
                cur.execute(
                    "UPDATE progress SET data = %s WHERE key = %s",
                    (json.dumps(found_data), found_row_key),
                )
            conn3.commit()
        finally:
            conn3.close()
    except Exception as e:
        return jsonify({"error": "db_save_failed", "detail": traceback.format_exc()[-500:]}), 500

    return jsonify({
        "topic_id": topic_id,
        "original_markers": original_markers,
        "new_markers": new_markers,
        "original_len": original_len,
        "new_len": len(marked),
        "pages": previews,
    })


def _deduplicate_notes(text):
    """Remove repeated content blocks from lecture notes, then re-insert page markers.

    Algorithm:
    1. Strip existing <<PAGE:N>> markers
    2. Split on slide-number patterns (e.g. "7 / 32") — reliable content boundaries
    3. Fingerprint each chunk (first 120 chars, whitespace-normalised)
    4. Drop chunks whose fingerprint has already been seen (keep first occurrence)
    5. Rejoin and re-run the page marker heuristic
    """
    import re

    # Strip existing markers
    clean = re.sub(r'<<PAGE:\d+>>\n?', '', text)

    # Split BEFORE each "N / M" slide-number pattern, keeping it at the start of its chunk.
    # re.split with a lookahead keeps the delimiter at the start of the next piece.
    raw_parts = re.split(r'(?=\b\d+\s*/\s*\d+\b)', clean)

    # Build chunks: a chunk is everything up to (but not including) the next slide number.
    # Small leading fragments (< 30 chars of non-whitespace) get prepended to the next chunk.
    chunks = []
    carry = ''
    for part in raw_parts:
        combined = carry + part
        carry = ''
        if len(combined.strip()) < 30:
            carry = combined
        else:
            chunks.append(combined)
    if carry.strip():
        if chunks:
            chunks[-1] += carry
        else:
            chunks.append(carry)

    # Fingerprint and deduplicate — keep first occurrence
    seen = set()
    deduped = []
    for chunk in chunks:
        fp = ' '.join(chunk.split())[:120]
        if not fp:
            continue
        if fp not in seen:
            seen.add(fp)
            deduped.append(chunk)

    rejoined = ''.join(deduped).strip()
    if not rejoined:
        return text  # safety: never return empty

    return _insert_page_markers_heuristic(rejoined)


@app.route("/admin/deduplicate-notes", methods=["POST"])
def admin_deduplicate_notes():
    """Deduplicate repeated content blocks in stored notes.

    Body (optional): {"topic_id": "fmi1"} — omit to process all topics.
    Returns per-topic stats: original_length, new_length, chunks_before,
    chunks_after, markers_placed.
    """
    import re, traceback
    admin_key = os.getenv("ADMIN_KEY", "studyai-admin")
    if request.headers.get("X-Admin-Key") != admin_key:
        return jsonify({"error": "Forbidden"}), 403

    body = request.get_json(silent=True) or {}
    filter_topic = body.get("topic_id")  # None = all topics

    try:
        conn = _get_db()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT key, data FROM progress")
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"error": "db_fetch_failed", "detail": traceback.format_exc()[-500:]}), 500

    results = []

    for db_key, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except Exception:
            continue

        notes = data.get("notes", {})
        if not notes:
            continue

        changed = False
        for topic_key, text in list(notes.items()):
            if filter_topic and topic_key != filter_topic:
                continue
            if not isinstance(text, str) or len(text.strip()) < 100:
                continue

            import re as _re
            orig_len = len(text)

            # Chunk count before (split same way as dedup fn for comparison)
            clean_for_count = _re.sub(r'<<PAGE:\d+>>\n?', '', text)
            chunks_before = max(1, len(_re.split(r'(?=\b\d+\s*/\s*\d+\b)', clean_for_count)))

            deduped = _deduplicate_notes(text)

            chunks_after = max(1, len(_re.split(r'(?=\b\d+\s*/\s*\d+\b)',
                                                 _re.sub(r'<<PAGE:\d+>>\n?', '', deduped))))
            markers_placed = len(_re.findall(r'<<PAGE:\d+>>', deduped))

            notes[topic_key] = deduped
            changed = True
            results.append({
                "topic_id": topic_key,
                "original_length": orig_len,
                "new_length": len(deduped),
                "chunks_before": chunks_before,
                "chunks_after": chunks_after,
                "markers_placed": markers_placed,
            })

        if changed:
            data["notes"] = notes
            conn2 = _get_db()
            try:
                with conn2.cursor() as cur:
                    cur.execute(
                        "UPDATE progress SET data = %s WHERE key = %s",
                        (json.dumps(data), db_key),
                    )
                conn2.commit()
            finally:
                conn2.close()

    if filter_topic and not results:
        return jsonify({"error": f"topic '{filter_topic}' not found"}), 404

    return jsonify({"topics": results, "count": len(results)})


@app.route("/admin/reextract-diagrams", methods=["POST"])
def admin_reextract_diagrams():
    """Retroactively identify diagrams and tables in stored notes using Gemini text analysis."""
    admin_key = os.getenv("ADMIN_KEY")
    if not admin_key:
        return jsonify({"error": "ADMIN_KEY not set on server"}), 403
    if request.headers.get("X-Admin-Key") != admin_key:
        return jsonify({"error": "Forbidden"}), 403
    if not google_client:
        return jsonify({"error": "Gemini not configured"}), 500

    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key, data FROM progress")
            rows = cur.fetchall()
    finally:
        conn.close()

    summary = {"rows_processed": 0, "topics_updated": [], "errors": []}

    for db_key, raw_data in rows:
        try:
            data = json.loads(raw_data)
        except Exception:
            continue

        notes = data.get("notes", {})
        if not notes:
            continue

        diagram_index = data.get("diagramIndex", {})
        changed = False

        for topic_key, text in list(notes.items()):
            if not isinstance(text, str) or len(text.strip()) < 100:
                continue
            try:
                prompt = (
                    "Identify any diagrams or tables described or referenced in these university lecture notes.\n\n"
                    f"{text[:20000]}\n\n"
                    "For each diagram or table found, return a JSON array:\n"
                    '[{"type": "diagram or table", "name": "e.g. IS-LM curve", "description": "economic meaning", "markdown": "if table: full markdown, else empty string"}]\n'
                    "If none found, return []. Return ONLY valid JSON array."
                )
                raw = ai_generate(prompt, system="You extract structured data from academic text.", max_tokens=2000, route='process_notes')
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[1] if '\n' in raw else raw[3:]
                if raw.endswith('```'):
                    raw = raw[:-3]
                items = json.loads(raw.strip())
                if items:
                    if topic_key not in diagram_index:
                        diagram_index[topic_key] = {}
                    for i, item in enumerate(items):
                        diagram_index[topic_key][f"retro_{i}"] = {
                            "type": item.get("name", "Diagram"),
                            "description": item.get("description", ""),
                            "image_url": "",
                            "markdown": item.get("markdown", "")
                        }
                    changed = True
                    summary["topics_updated"].append(topic_key)
            except Exception as e:
                summary["errors"].append(f"{topic_key}: {str(e)[:80]}")

        if changed:
            data["diagramIndex"] = diagram_index
            conn2 = _get_db()
            try:
                with conn2.cursor() as cur:
                    cur.execute(
                        "UPDATE progress SET data = %s WHERE key = %s",
                        (json.dumps(data), db_key),
                    )
                conn2.commit()
            finally:
                conn2.close()
            summary["rows_processed"] += 1

    return jsonify(summary)


_ECON_TERMS = {
    'demand','supply','market','price','rate','return','risk','value','model',
    'theory','regression','coefficient','variable','function','curve',
    'equilibrium','elasticity','utility','profit','cost','revenue','output',
    'labour','labor','capital','interest','inflation','gdp','gnp','ols',
    'bond','equity','asset','portfolio','yield','debt','deficit','fiscal',
    'monetary','aggregate','sector','trade','exchange','reserve','liquidity',
    'leverage','default','credit','spread','dividend','earnings','volatility',
    'variance','correlation','estimate','deviation','mean','median',
    'hypothesis','statistic','probability','distribution','derivative',
    'marginal','average','nominal','real','growth','income','wage','tax',
    'subsidy','index','ratio','shock','policy','bank','money','stock','flow',
    'firm','consumer','producer','welfare','optimal','constraint','budget',
    'preference','monopoly','oligopoly','competition','efficiency',
    'externality','endogenous','exogenous','parameter','intercept','slope',
    'error','residual','forecast','trend','multiplier','velocity','solvency',
    'beta','alpha','gamma','sigma','epsilon','theta','lambda','delta',
    'omega','rho','phi','mu',
}
_MATH_CHARS = set('=+*/^~≈≠≤≥<>∑∫∂∇')
_GREEK_CHARS = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')


def _line_has_academic_content(line):
    if any(c in _MATH_CHARS or c in _GREEK_CHARS for c in line):
        return True
    words = re.findall(r'[a-zA-Z]+', line.lower())
    return any(w in _ECON_TERMS for w in words)


def clean_extracted_text(text):
    from collections import Counter
    lines = text.split('\n')
    counts = Counter(l.strip() for l in lines if l.strip())
    removed = []
    out = []
    for line in lines:
        t = line.strip()
        if t and counts[t] > 5 and len(t) < 60 and not _line_has_academic_content(t):
            removed.append(t)
        else:
            out.append(line)
    if removed:
        unique = list(dict.fromkeys(removed))
        print(f'[clean_extracted_text] Removed {len(removed)} lines ({len(unique)} unique): {unique}')
    return '\n'.join(out)


@app.route("/extract", methods=["POST"])
def extract():
    """Extract text from an uploaded file. PDF: full Gemini Vision pipeline. DOCX: text only.
    Zero Claude/Anthropic API calls during extraction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = file.read()
    filename = file.filename or 'document'

    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    if filename.lower().endswith('.docx'):
        raw_text = extract_docx_text(file_bytes)
        result = {
            "text": clean_extracted_text(raw_text),
            "raw_text": raw_text,
            "diagrams": [],
            "tables": [],
            "has_enhanced": False
        }
    else:
        result = extract_pdf_full(file_bytes, filename)

    if not result.get("text") or len(result["text"].strip()) < 50:
        return jsonify({"error": "Could not extract text from this file. Try saving as .txt instead."}), 422

    return jsonify({
        "text": result["text"],
        "raw_text": result["raw_text"],
        "words": len(result["text"].split()),
        "pages": len(result["text"].split("\n")),
        "diagrams": result.get("diagrams", []),
        "tables": result.get("tables", []),
        "has_enhanced": result.get("has_enhanced", False)
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
        "You are a specialist academic tutor extracting structured study content from university lecture notes "
        "for a 2nd year economics and finance student at the University of Birmingham. "
        "Your output will be shown directly in a study app — it must be dense, precise, and exam-ready. "
        "Use the lecturer's exact notation and terminology throughout. "
        "Every field must be filled with content specific to this topic — nothing generic, nothing vague. "
        "Identify how each concept connects to others in the course: these connections help students see the bigger picture."
    )

    prompt = (
        f"Extract structured study content from these lecture notes on '{topic_name}':\n\n"
        f"{text[:50000]}\n\n"
        "Return ONLY a valid JSON object, no markdown fences:\n"
        '{"key_concepts":[{"term":"...","definition":"one precise sentence using lecturer\'s exact language","formula":"exact notation if applicable, else empty string"}],'
        '"formulas":[{"name":"...","formula":"exact notation","variables":"what each symbol means","when_used":"one sentence on when/why you apply this"}],'
        '"exam_points":["specific examinable point 1","specific examinable point 2"],'
        '"common_mistakes":["specific error students make 1","specific error students make 2"],'
        '"connections":["This topic links to <other topic> because <reason>","..."]}\n\n'
        "Rules:\n"
        "- key_concepts: 5-8 most important concepts. Definition must be the precise academic definition.\n"
        "- formulas: only actual mathematical or statistical formulas. Include every variable.\n"
        "- exam_points: 4-7 points that are likely to appear on exams. Must be specific to this topic.\n"
        "- common_mistakes: 2-4 errors specific to this topic — not generic exam advice.\n"
        "- connections: 2-4 links to other topics or concepts in Money, Banking and Finance. Each must explain WHY they connect.\n"
        "- If a section genuinely has nothing relevant, return an empty array."
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
    """Save progress for a user. Auto-deduplicates notes on every save."""
    import re as _re
    key = request.args.get("key", "default")
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    # Auto-dedup: if the payload contains notes, clean each topic before saving.
    # Only triggers when the text is long enough to plausibly contain duplicates
    # and has a slide-number pattern (N / M) that our dedup relies on.
    notes = data.get("notes")
    if isinstance(notes, dict):
        for topic_key, text in list(notes.items()):
            if (isinstance(text, str)
                    and len(text) > 5000
                    and _re.search(r'\b\d+\s*/\s*\d+\b', text)):
                try:
                    notes[topic_key] = _deduplicate_notes(text)
                except Exception:
                    pass  # never let dedup break a save

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
