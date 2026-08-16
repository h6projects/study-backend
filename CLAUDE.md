# Product Vision

StudyAI is a guided learning system for university students — not a chatbot, not a research assistant, not a NotebookLM clone.

## Core philosophy
- NotebookLM helps users understand content on demand — StudyAI guides users through learning step by step
- The workflow is structured: modules → topics → lessons → quizzes → mastery
- Every feature should move a student from "I have lecture slides" to "I understand this topic and can answer exam questions"

## What this app does
- Generates structured lessons for each topic from uploaded lecture materials
- Teaches concepts clearly and progressively with examples and common mistakes
- Generates quizzes to test understanding at increasing difficulty
- Tracks progress and mastery per topic
- Identifies weak areas and guides what to study next

## What this app does NOT do
- Does not default to chatbot-style interactions
- Does not treat users as researchers exploring content freely
- Does not contradict uploaded lecture materials
- Does not produce vague or generic outputs

## Content priority
- Uploaded lecture materials are always the primary source
- General knowledge only used to clarify or enhance — never to replace lecturer content

## Tone
- Personal tutor guiding a student through their course
- Clear, structured, exam-focused
- Rigorous but never obscure

---

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
1. Home (screen-home) — module selector, topic list, exam countdown bar, weak spots panel, session plan panel, file upload, past papers
2. Notes (screen-notes) — lecture notes read/edit tabs, summary/full toggle, PROCESS bar, badges (SUMMARISED, ENHANCED), exercise buttons
3. Lesson (screen-lesson) — 6-8 slides generated from notes, exam mode toggle on last slide
4. Results (screen-results) — score, knowledge gaps, spaced repetition scheduling
5. Paper (screen-paper) — past paper practice with AI marking
6. Notes browser (screen-notes-browser) — all saved notes across modules
7. Flashcards (screen-flashcards) — card-flip exercises
8. Fill blanks (screen-fill-blanks) — cloze exercises
9. Speed round (screen-speed) — timed 60s quiz across all module topics
10. Processing (screen-processing) — batch file upload progress bar
11. Batch assign (screen-batch-assign) — review AI-assigned topic assignments before saving

## State persisted in PostgreSQL
- progress (completed topics)
- notes (lecture notes per topic, cap 50000 chars)
- outlines (module outlines per module, cap 50000 chars)
- customTopics (AI-extracted topics from outlines)
- markSchemes (mark scheme text per topic)
- papers (uploaded past papers)
- reviews (spaced repetition schedule: {nextReview, lastScore, intervalDays})
- mastery (mastery score 0-100 per topic)
- xp (total XP points)
- quizzesPassed
- processedFiles (uploaded filenames per module)
- noteMeta ({wordCount, fileCount, files[], summarised} per topic)
- processedNotes (AI-processed concept/formula/exam data per topic)
- customHints (user-set topic hints)
- rawNotes (original extraction text before merging)
- cleanedNotes (canonical first-upload text)
- geminiNotes (enhanced extraction text from vision pipeline)
- weakSpots ({topicId: {concept: {wrong, total, lastSeen}}})
- examDates ([{name, date, mod, notes}])
- diagramIndex ({topicId: {pageNum: {type, description, image_url, svg_hint}, 't'+pageNum: {type, image_url, markdown}}})

## State NOT persisted (session-only)
- screenHistory (back navigation stack, session only)
- currentMod, currentTopic (active selection)
- slides, slideIdx, quizQuestions, quizIdx, quizAnswers, phase (lesson/quiz in progress)
- batchItems (files being processed)
- _batchGeminiReprocess (force re-extract flag)

## Key design decisions
- No browser-side AI calls — everything goes through backend
- No login system — single user identified by localStorage key
- Hardcoded topics are the fallback — customTopics override them per module
- uk, fmi, cf, eco modules force hardcoded topics (delete customTopics on load)
- Notes cap 50000 chars everywhere
- Spaced repetition intervals: 1/3/7/14 days based on score
- Back navigation: showScreen() pushes to state.screenHistory; goBack() pops; Escape key also triggers goBack()
- Weak spot threshold: >40% wrong rate with >=3 attempts; dismissed per-concept in localStorage for 24h
- Session time estimates: urgent=20min, long notes (>2000 words)=20min, medium (>500)=15min, short=10min

## PDF extraction pipeline
1. `POST /extract` — pymupdf extracts text; Gemini Vision processes each page for diagrams/tables
2. Text output has `<<PAGE:N>>` markers before each page's text block
3. Diagrams/tables stored in Supabase Storage bucket 'diagrams', URLs returned
4. Frontend stores URLs in `state.diagramIndex[topicId][pageNum]` (diagrams) and `state.diagramIndex[topicId]['t'+pageNum]` (tables)
5. `renderNotesFormatted()` splits on `<<PAGE:N>>` markers, injects diagram/table images after each page's text
6. Safety-net section "DIAGRAMS & TABLES" appended for any unrendered entries (old notes without markers)
7. ENHANCED badge shown in notes screen when geminiNotes present for a topic

## AI branding policy
- Never show provider names (Gemini, Claude, Anthropic) in user-facing UI
- Use neutral language: "AI", "enhanced extraction", "processed"
- The ENHANCED badge/tag indicates Gemini Vision was used — label stays as "ENHANCED" not "GEMINI VISION"
- Internal code variables (geminiNotes, callClaude) may keep provider names — only UI-facing text matters

## What NOT to do
- Never add browser-side API calls to Anthropic or Gemini
- Never store sensitive credentials in code
- Never rewrite whole files unless explicitly asked
- Never remove the hardcoded topic fallbacks
- Never break the PostgreSQL progress save/load cycle
- Never introduce async/await in non-async functions
- Never truncate notes below 50000 chars
- Never use /tmp for storage — always PostgreSQL
- Never add Fraunces serif except .slide h3 (lesson slides) and .exam-q (past papers)
- Never use border-radius above 6px — target is 4px, no pills

## Batched polling pattern for heavy AI operations

For any backend operation that takes >30 seconds:
- Use a `POST /route/start` → returns `session_id` immediately
- Use `GET /route/status?session_id=X` polled every 2s from frontend
- Store session state in a module-level dict (`_sessions`) with 10-minute TTL — never in PostgreSQL
- Background work runs in `threading.Thread(daemon=True)`
- Frontend starts showing results as soon as first batch is ready; user doesn't wait for all batches

**Already implemented:** `/topic-practice/start` + `/topic-practice/status` (3 batches of 4 questions)

**Candidates for the same pattern (not yet implemented):**
- `/parse-paper` — past paper question extraction (can be slow for long PDFs)
- `/extract` with vision — page-by-page streaming would improve UX
- `/lesson` — 8 slides could stream one at a time (~20s current wait, lower priority)

## AI Model Strategy

The app supports multiple AI providers. Do not hardcode around a single model.

Current: Claude (claude-sonnet-4-20250514) for quality-critical tasks (lesson, quiz, flashcards, fill-blanks, mark-answer, parse-paper).
Gemini (gemini-2.0-flash) for cheap/long-doc tasks (summarise, process-notes, sort, extract-topics).

Never use model-specific quirks as a feature dependency. Always route through the abstraction layer.

All generation calls go through `ai_generate(prompt, system, max_tokens, model)` in app.py.
Vision calls go through `ai_vision(image_data, prompt)`.
Switch providers at runtime with the `AI_PROVIDER` environment variable (default: `claude`, future: `gemini`).

## Current Railway URL
https://study-backend-production-eb16.up.railway.app

## Railway environment variables
- `ANTHROPIC_API_KEY` — Anthropic API key (required)
- `DATABASE_URL` — PostgreSQL connection string via Supabase pooler (required)
- `AI_PROVIDER` — AI provider to use: `claude` (default) or `gemini` (future). Switch without code changes.
- `PORT` — server port (Railway sets this automatically)

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

## Supabase persistence protection

### Keepalive (.github/workflows/keepalive.yml)
- Runs every 3 days (cron `0 9 */3 * *`) and on `workflow_dispatch`
- Calls GET /healthz on the Railway backend, which executes `SELECT 1` against Supabase
- This registers as DB activity, preventing Supabase free-tier auto-pause (triggers after ~7 days of inactivity)
- Workflow fails visibly if /healthz returns non-200, giving an alert that the backend or DB is down
- To trigger manually: GitHub → Actions → "Supabase keepalive ping" → Run workflow

### Weekly backup (.github/workflows/backup.yml)
- Runs every Sunday at 09:00 UTC (cron `0 9 * * 0`) and on `workflow_dispatch`
- Uses `pg_dump --table=progress --data-only` to dump only the user state table, gzipped
- Artifact named `progress-YYYY-MM-DD.sql.gz`, retained for 30 days
- Fails if the dump is under 50 bytes (empty or connection failed)
- **Requires GitHub secret**: `DATABASE_URL` — set it at:
  GitHub repo → Settings → Secrets and variables → Actions → New repository secret
  Name: `DATABASE_URL`, Value: copy from Railway dashboard → StudyApp service → Variables → DATABASE_URL
- To trigger manually: GitHub → Actions → "Weekly Postgres backup" → Run workflow

### Restoring from a backup artifact
1. Download the `.sql.gz` artifact from GitHub → Actions → the backup run → Artifacts
2. Decompress: `gunzip progress-YYYY-MM-DD.sql.gz`
3. The file contains only `INSERT` statements for the `progress` table (no schema)
4. If restoring to an existing project (table already exists): `psql "$DATABASE_URL" < progress-YYYY-MM-DD.sql`
5. If restoring to a fresh Supabase project: create the table first (schema is in app.py `_init_db()`), then restore:
   `python app.py` once with a valid DATABASE_URL will run `_init_db()` and create the schema; then `psql < file.sql`
6. Verify: `psql "$DATABASE_URL" -c "SELECT user_id, length(data) FROM progress;"`

## Cost awareness
- Claude Code terminal usage is covered by Claude Pro subscription — no extra billing
- Anthropic API credits are only used when the app calls Claude (lessons, quizzes, sorting)
- Keep backend prompts efficient — avoid unnecessary large context unless quality requires it
- The /sort route uses max_tokens 20 — keep it minimal
- Lesson and quiz prompts use up to 50000 chars of notes — this is intentional for quality

## Post-exams technical debt

### Move Deep Practice session state to Supabase (multi-worker safety)

**Current mitigation:** Procfile uses `--workers 1 --preload` so all requests hit a single process. `_dp_cache` in-memory dict is safe within one process. This breaks if Railway ever scales horizontally or during rolling deploys.

**Proper fix:** Replace `_dp_cache` + `dp_sessions` PostgreSQL table with a `practice_sessions` table that both workers read/write atomically.

Proposed schema:
```sql
CREATE TABLE practice_sessions (
    session_id   TEXT PRIMARY KEY,
    status       TEXT NOT NULL DEFAULT 'generating',  -- generating | complete | failed | partial
    batches_ready INTEGER NOT NULL DEFAULT 0,
    batches_total INTEGER NOT NULL DEFAULT 3,
    questions    JSONB NOT NULL DEFAULT '[]',
    error        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON practice_sessions (created_at);
```

Implementation notes:
- Drop `_dp_cache` dict entirely; all reads/writes go to `practice_sessions` table
- Use `SELECT ... FOR UPDATE SKIP LOCKED` for atomic batch appends to the `questions` JSONB column
- Cleanup: delete rows older than 30 min (not 10 min) to be safe across slow deploys
- Once this is done, remove `--workers 1` from Procfile and let Railway scale normally
- `/topic-practice/status` will then tolerate any number of replicas

**Why deferred:** Railway free tier rarely spins up multiple replicas; `--workers 1` is sufficient during exams. This becomes urgent only if response times degrade under load.
