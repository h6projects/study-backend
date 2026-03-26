# StudyAI — Full Project Context

## What this is
An AI-powered study app for a 2nd year Money, Banking and Finance student at the University of Birmingham. Built to help with exam prep using the student's own lecture notes.

## Tech stack
- Frontend: studyapp.html — single HTML file, no framework, vanilla JS
- Backend: app.py — Python Flask, hosted on Railway
- Database: PostgreSQL via Supabase (connection pooler port 6543)
- AI: Anthropic API (claude-sonnet-4-20250514) — all calls server-side via backend
- Hosting: Railway (backend), local file or future Netlify (frontend)
- Version control: GitHub, auto-deploys to Railway on push

## Four modules
1. Financial Markets & Institutions (fmi) — 10 topics, weeks 1-10, two lecturers
2. Corporate Finance (cf) — 8 topics
3. Econometrics (eco) — 9 topics, Dr Ercolani weeks 1-5, Dr Melander weeks 7-10
4. Contemporary Issues in the UK Economy (uk) — 8 topics

## Backend routes
- POST /extract — PDF to text (uses pymupdf + Claude Vision for diagram pages, pypdf fallback)
- POST /summarise — compresses long docs >40k chars via chunked Claude summarisation
- POST /lesson — generates 6-8 slides from notes
- POST /quiz — generates 6 questions, learn or exam mode
- POST /sort — matches text to topic indices
- POST /extract-topics — extracts topic list from outline
- POST /parse-paper — extracts questions from past paper
- POST /mark-answer — AI examiner marks free text answer
- POST /clear-custom-topics — removes bad extracted topics
- GET/POST /progress — PostgreSQL persistence for all state
- GET /healthz — health check

## Frontend screens
1. Home — module selector, topic list, file upload, outline upload, past papers, spaced repetition widget
2. Notes — lecture notes with read/edit tabs, formatted preview, Greek symbols
3. Lesson — 6-8 slides generated from notes, exam mode toggle on last slide
4. Results — score, knowledge gaps, spaced repetition scheduling
5. Paper — past paper practice with AI marking
6. Notes browser — all saved notes across modules
7. Exam mode — module-level exam with questions spanning all topics

## State persisted in PostgreSQL
- progress (completed topics)
- notes (lecture notes per topic, cap 50000 chars)
- outlines (module outlines per module, cap 50000 chars)
- customTopics (AI-extracted topics from outlines)
- papers (uploaded past papers)
- reviews (spaced repetition schedule)
- quizzesPassed

## Key design decisions
- No browser-side Anthropic calls — everything goes through backend
- No login system — single user identified by localStorage key
- Hardcoded topics are the fallback — customTopics override them per module
- uk and eco modules force hardcoded topics (delete customTopics on load)
- Notes cap 50000 chars everywhere
- Spaced repetition intervals: 1/3/7/14 days based on score

## What NOT to do
- Never add browser-side API calls to Anthropic
- Never store sensitive credentials in code
- Never rewrite whole files unless explicitly asked
- Never remove the hardcoded topic fallbacks
- Never break the PostgreSQL progress save/load cycle
- Never introduce async/await in non-async functions
- Never truncate notes below 50000 chars
- Never use /tmp for storage — always PostgreSQL
- Never add Fraunces serif except .slide h3 (lesson slides) and .exam-q (past papers)
- Never use border-radius above 6px — target is 4px, no pills

## Current Railway URL
https://study-backend-production-eb16.up.railway.app

## Workflow
- All changes committed and pushed to GitHub on main branch
- Railway auto-deploys on push
- Always commit after each change with a clear message
- Read files before editing — never rewrite unless necessary
- Batch all related changes into one commit
- Make all changes to the same file in one pass — read once, edit once
- When fixing multiple issues, handle them all in one response
- Ask all clarifying questions upfront rather than back and forth
- Prefer targeted edits over full rewrites
- Check what already exists before writing new code
- Prioritise high-impact changes when usage is limited

## Cost awareness
- Claude Code terminal usage is covered by Claude Pro subscription — no extra billing
- Anthropic API credits are only used when the app calls Claude (lessons, quizzes, sorting)
- Keep backend prompts efficient — avoid unnecessary large context unless quality requires it
- The /sort route uses max_tokens 20 — keep it minimal
- Lesson and quiz prompts use up to 50000 chars of notes — this is intentional for quality
