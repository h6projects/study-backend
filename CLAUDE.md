# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

StudyAI is a study tool for a Money, Banking & Finance student at the University of Birmingham. It generates AI-powered lessons and quizzes from uploaded lecture notes.

- **Backend:** `app.py` — Flask, deployed on Railway via GitHub auto-deploy, uses Anthropic API server-side
- **Frontend:** `studyapp.html` — single HTML file (no framework, no build step), opened directly in a browser
- No browser-side Anthropic calls — all AI goes through the backend

## Running locally

```bash
pip install -r requirements.txt
ANTHROPIC_API_KEY=your_key python app.py   # runs on port 5001
```

Open `studyapp.html` directly in a browser. It auto-detects `localhost:5001` when opened as a local file or from localhost, and uses the Railway URL in production.

Test the Claude connection: `GET http://localhost:5001/debug_api`

## Architecture

### Backend routes (`app.py`)

| Route | Method | Purpose |
|-------|--------|---------|
| `/extract` | POST | Upload PDF → extract text (pypdf, max 40 pages) |
| `/sort` | POST | Classify text against a topic list → returns comma-separated topic indices |
| `/lesson` | POST | Generate a 4-slide lesson as JSON (`{title, key_concepts, slides, exam_tips}`) |
| `/quiz` | POST | Generate 4 MCQs as JSON (`[{question, options, correct, explanation, concept}]`) |
| `/progress` | GET/POST | Load/save user state; stored as `/tmp/progress_<md5>.json` keyed by `?key=` param |

All Claude calls use `claude-sonnet-4-20250514`. Responses are plain text or JSON; the helper `_message_text()` extracts text from content blocks, and lesson/quiz routes strip markdown fences before `json.loads()`.

`TOPIC_CONTEXT` is a hardcoded dict in `app.py` that enriches topic names with keyword hints before the `/sort` prompt, improving classification accuracy.

### Frontend state & screens (`studyapp.html`)

Four screens driven by `showScreen(id)`:
- **home** — module/topic list + file upload drop zone
- **notes** — review/paste notes before starting a lesson
- **lesson** — slides phase then quiz phase, rendered into `#lesson-content`
- **results** — score, knowledge gaps, per-question review

`state` is a plain JS object holding progress, notes, outlines, current topic/module, slide/quiz position.

### Hardcoded content

- **4 modules, 24 topics** — Financial Markets & Institutions, Corporate Finance, Econometrics, Contemporary UK Economy — defined in the `MODULES` array
- **Pre-filled notes** — `eco1` (OLS regression) and `eco2` (hypothesis testing) have Ercolani lecture notes baked into `PREFILLED`
- **Pre-loaded outline** — the Econometrics module outline is in `PRELOADED_OUTLINES` (merged with any user-uploaded outlines on load)

### File upload flow

1. User drops PDF/TXT/MD onto the drop zone
2. Frontend calls `/extract` for PDFs, or reads text directly for TXT/MD
3. Text is sent to `/sort` with the current module's topic names
4. Matched topics receive the file text appended to `state.notes[topicId]`
5. Progress is saved via `/progress`

Module outlines (uploaded per module in the sidebar) are stored in `state.outlines[modId]` and passed as context to `/lesson` calls.

### Progress persistence

Progress is saved to `/progress?key=studyai_user` as `{progress, quizzesPassed, notes, outlines}`. Notes are capped at 8000 chars per topic before saving. Pre-filled notes (`PREFILLED`) are never overwritten by saved notes on load.

A topic is marked `done` when the user scores ≥ 75% on the quiz.

## Deployment

Deployed on Railway. GitHub push to `main` triggers auto-deploy. The Railway backend URL is hardcoded in `studyapp.html`:
```js
'https://study-backend-production-be20.up.railway.app'
```
`ANTHROPIC_API_KEY` must be set as a Railway environment variable.
