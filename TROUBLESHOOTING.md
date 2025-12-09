# 🚨 TROUBLESHOOTING GUIDE - GRAIL-LM Won't Start

## Current Issues Detected

### Issue 1: Python Virtual Environment Broken ❌
**Problem**: `.venv311` points to non-existent Python installation  
**Solution**: Recreate virtual environment

```powershell
# Remove broken venv
Remove-Item -Recurse -Force .venv311

# Find your Python installation
where.exe python

# Create new venv with correct Python
python -m venv .venv

# Activate
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Issue 2: Docker Desktop Not Running ❌
**Problem**: Docker daemon is not accessible  
**Solution**: Start Docker Desktop

1. Press `Windows Key`
2. Search for "Docker Desktop"
3. Launch it
4. Wait for "Docker Desktop is running" in system tray
5. Try again: `docker-compose up`

---

## 🎯 SIMPLIFIED LAUNCH (Choose ONE Method)

### Option A: Docker-Only (Recommended - No Python Needed!)

**Prerequisites**: Only Docker Desktop running

```powershell
# 1. Start Docker Desktop first!
# 2. Build and run everything in containers
docker-compose up --build

# Services available at:
# - API: http://localhost:8010/health
# - Neo4j: http://localhost:7474 (neo4j/password)
# - Streamlit: http://localhost:8501
```

**Note**: This runs everything inside Docker, including Python/API/UI.

---

### Option B: Local Python (If You Fix Python First)

**Prerequisites**: Working Python 3.11+

```powershell
# 1. Create fresh venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start just Docker services (Neo4j + Ollama)
docker-compose up neo4j ollama -d

# 4. Wait 30 seconds
Start-Sleep -Seconds 30

# 5. Load data (requires neo4j package)
python scripts\load_neo4j.py

# 6. Start API locally
uvicorn backend.api:app --reload --port 8001

# 7. Start Streamlit (in new terminal)
streamlit run app\streamlit_app.py --server.port 8501
```

---

## 🔥 QUICK FIX: Just Start Docker Services

If you just want to see if Neo4j and Ollama work:

```powershell
# 1. Start Docker Desktop

# 2. Run only infrastructure
docker-compose up neo4j ollama -d

# 3. Wait 30 seconds
Start-Sleep -Seconds 30

# 4. Check Neo4j
Start-Process "http://localhost:7474"
# Login: neo4j / password

# 5. Check Ollama
curl http://localhost:11434/api/tags
```

---

## ⚡ MINIMAL WORKING DEMO (Without Full Setup)

Can't get anything to work? Use the existing backend code without Neo4j/GNN:

```powershell
# 1. Fix Python first
where.exe python
# Use that path:
C:\Path\To\Python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install minimal deps
pip install fastapi uvicorn streamlit pandas networkx

# 3. Disable Neo4j/GNN in code temporarily
$env:USE_NEO4J="false"
$env:USE_GNN="false"

# 4. Run API (uses in-memory graph)
uvicorn backend.api:app --reload --port 8001

# 5. Run Streamlit (in new terminal)
streamlit run app\streamlit_app.py
```

---

## 🩺 Diagnostic Commands

Run these to see what's actually wrong:

```powershell
# Check Python
where.exe python
python --version

# Check Docker
docker --version
docker ps

# Check if Docker Desktop is running
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue

# Check ports are free
netstat -an | findstr "8001 8010 7474 7687 11434"

# Check existing containers
docker ps -a
```

---

## 🎯 FASTEST PATH TO WORKING SYSTEM

**If Docker Desktop is installed:**

1. **Start Docker Desktop** (wait for green icon in system tray)
2. Run:
   ```powershell
   docker-compose up --build
   ```
3. Wait 2 minutes for all services to start
4. Open: http://localhost:8501 (Streamlit)
5. Try asking: "What companies does Apple partner with?"

**That's it!** Everything runs in containers, no Python issues.

---

## 🆘 Still Not Working?

**Check Docker Desktop**:
- Is it actually running? (Green whale icon in system tray)
- Is WSL 2 enabled? (Docker Desktop → Settings → General)
- Try restarting Docker Desktop

**Python Still Broken?**:
- Download fresh Python 3.11 from python.org
- Install for all users
- Add to PATH during installation
- Restart PowerShell
- Try creating venv again

**Need Help?**:
1. Run: `docker-compose logs` and share errors
2. Check: Docker Desktop → Troubleshoot → Clean/Purge data
3. Restart your computer (fixes many Docker issues!)
