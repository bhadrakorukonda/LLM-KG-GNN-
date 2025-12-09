# GRAIL-LM Setup Script for Windows
# Run with: .\scripts\setup.ps1

Write-Host "🚀 GRAIL-LM Setup Script" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "✓ Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}
Write-Host "  $pythonVersion" -ForegroundColor Green

# Check Docker
Write-Host "✓ Checking Docker..." -ForegroundColor Yellow
$dockerVersion = docker --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Docker not found. Docker is required for Neo4j and Ollama." -ForegroundColor Yellow
    Write-Host "  Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
} else {
    Write-Host "  $dockerVersion" -ForegroundColor Green
}

# Create virtual environment
Write-Host ""
Write-Host "✓ Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "  Virtual environment already exists" -ForegroundColor Gray
} else {
    python -m venv .venv
    Write-Host "  Created .venv/" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host ""
Write-Host "✓ Installing dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --quiet
Write-Host "  Dependencies installed" -ForegroundColor Green

# Create .env file
Write-Host ""
Write-Host "✓ Setting up environment..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  .env already exists" -ForegroundColor Gray
} else {
    Copy-Item .env.example .env
    Write-Host "  Created .env from .env.example" -ForegroundColor Green
}

# Create models directory
Write-Host ""
Write-Host "✓ Creating directories..." -ForegroundColor Yellow
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
    Write-Host "  Created models/" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start services:     docker-compose up -d" -ForegroundColor White
Write-Host "  2. Load Neo4j data:    python scripts/load_neo4j.py" -ForegroundColor White
Write-Host "  3. Train GNN (opt):    python scripts/train_gnn.py" -ForegroundColor White
Write-Host "  4. Run Streamlit:      streamlit run app/streamlit_app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or use the API directly:" -ForegroundColor Cyan
Write-Host "  uvicorn backend.api:app --reload --port 8001" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: README.md" -ForegroundColor Gray
