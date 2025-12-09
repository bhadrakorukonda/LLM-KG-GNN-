# Fix Python Environment and Launch GRAIL-LM

Write-Host "🔧 Fixing Python Environment..." -ForegroundColor Cyan

# Deactivate current broken venv
deactivate 2>$null

# Remove broken venv
if (Test-Path ".venv311") {
    Write-Host "Removing broken .venv311..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv311
}

# Create new venv with correct Python
Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
C:\Users\bhadr\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv

# Activate new venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "Installing dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "✅ Python environment fixed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start Docker Desktop" -ForegroundColor White
Write-Host "2. Run: docker-compose up --build" -ForegroundColor White
Write-Host ""
