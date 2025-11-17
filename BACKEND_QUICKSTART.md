# ORION Backend - Quick Start Guide

## 🚀 Start the Backend API

### Option 1: Quick Start with Mock Data (Recommended for Testing Frontend) ✅

Start the simplified development server with mock data - no database required!

```bash
# From project root
cd src/api
../../venv/bin/python -m uvicorn app_dev:app --reload --host 0.0.0.0 --port 8000
```

**This option:**
- ✅ Starts immediately - no setup needed
- ✅ Returns mock data for all endpoints
- ✅ Full CORS support for localhost:3000, localhost:3001
- ✅ Perfect for frontend development and testing
- ✅ No PostgreSQL, Redis, or other services needed

### Option 2: Basic Development Server (With Production App)

Start the FastAPI backend server with minimal dependencies:

```bash
# From project root
cd src/api
python -m uvicorn app:app --reload --port 8000
```

**Note:** This will show connection errors for PostgreSQL, Redis, etc., and may fail to start due to missing dependencies.

### Option 2: Full Stack with Docker (Production-like)

```bash
# Start all services (PostgreSQL, Redis, Neo4j, etc.)
docker-compose up -d

# Start the API server
cd src/api
python -m uvicorn app:app --reload --port 8000
```

### Option 3: Development with Mock Services

```bash
# Set environment to use mock services
export ORION_USE_MOCKS=true

# Start the API
cd src/api
python -m uvicorn app:app --reload --port 8000
```

---

## ✅ Verify Backend is Running

Once started, you should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

Test the API:
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

---

## 🔌 Connect Frontend to Backend

The frontend is already configured to connect to `http://localhost:8000`.

**CORS is now configured to allow:**
- `http://localhost:3000` (default Next.js port)
- `http://localhost:3001` (current frontend port)
- `http://localhost:8000` (API self-reference)

Once the backend is running, refresh the frontend at http://localhost:3001 and the CORS errors will disappear!

---

## 📝 Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# Minimal configuration for development
ENVIRONMENT=development
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:8000

# Optional: Database (if you have PostgreSQL running)
DATABASE_URL=postgresql+asyncpg://orion:password@localhost:5432/orion_db

# Optional: Redis (if you have Redis running)
REDIS_URL=redis://localhost:6379/0
```

---

## 🛠️ Dependencies

### Required Python Packages

If you get import errors, install backend dependencies:

```bash
cd src/api
pip install -r ../../requirements.txt

# Or minimal install:
pip install fastapi uvicorn pydantic python-dotenv
```

### Optional Services

**PostgreSQL** (for database)
```bash
# macOS
brew install postgresql
brew services start postgresql

# Create database
createdb orion_db
```

**Redis** (for caching)
```bash
# macOS
brew install redis
brew services start redis
```

---

## 🔍 Troubleshooting

### "Module not found" errors
```bash
# Install all requirements
pip install -r requirements.txt
```

### "Database connection failed"
The API will start anyway, but some endpoints won't work. You can either:
1. Start PostgreSQL (see above)
2. Use mock mode: `export ORION_USE_MOCKS=true`
3. Ignore the errors (frontend UI will still work)

### CORS errors persist
1. Make sure backend is running on port 8000
2. Check `src/api/config.py` includes `http://localhost:3001`
3. Restart the backend server after changes

---

## 📊 What Works Without Services

Even without PostgreSQL/Redis/Neo4j, these endpoints work:
- ✅ `/health` - Health check
- ✅ `/docs` - API documentation
- ✅ `/api/v1/*` - Most GET endpoints (with mock data)

These require services:
- ⏸️ Structure upload (needs PostgreSQL)
- ⏸️ Simulation jobs (needs Celery + Redis)
- ⏸️ Knowledge graph (needs Neo4j)

---

## 🎯 Next Steps

1. **Start the backend:**
   ```bash
   cd src/api
   python -m uvicorn app:app --reload --port 8000
   ```

2. **Keep frontend running:**
   ```bash
   # Already running at http://localhost:3001
   ```

3. **Test the connection:**
   - Open http://localhost:3001
   - Navigate to "Structures" or "Design Search"
   - CORS errors should be gone!
   - You'll see loading states or "No data" messages (expected without database)

4. **Add real data (optional):**
   - Start PostgreSQL
   - Run migrations: `alembic upgrade head`
   - Seed data: `python scripts/seed_workflows.py`

---

## 🚀 Ready!

Your ORION platform is ready for development!
- **Frontend:** http://localhost:3001 ✅ Running
- **Backend:** http://localhost:8000 ⏸️ Start with command above
- **Docs:** http://localhost:8000/docs (once backend starts)
