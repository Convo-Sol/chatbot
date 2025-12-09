# TODO for Flask App Optimization

- [x] Update Procfile with optimized Gunicorn command (2 workers, 4 threads, gthread, 120s timeout, /dev/shm, preload, log to stdout)
- [x] Optimize retrieval.py: Move ChromaDB initialization to lazy-loaded function to avoid blocking startup
- [x] Add health check endpoint (/health) in app.py for Render health checks
- [x] Test locally with gunicorn (N/A on Windows; test on Render)
- [ ] Deploy to Render and monitor logs
