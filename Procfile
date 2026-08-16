# workers=1 required until practice session state is moved to a shared store (Supabase).
# See backlog: "move Deep Practice session state to Supabase for multi-worker safety".
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --preload --timeout 300 --log-level info --access-logfile - --error-logfile -
