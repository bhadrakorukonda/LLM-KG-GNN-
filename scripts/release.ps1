$ErrorActionPreference = "Stop"

param(
  [string]$Version = "0.1.0"
)

Write-Host "Tagging v$Version..."
git tag v$Version -m "v$Version"
git push origin v$Version

Write-Host "Release checklist:"
@(
  "- Verify CHANGELOG.md entries",
  "- Ensure CI green (pytest + ruff)",
  "- Publish Docker images if applicable",
  "- Verify /version returns $Version",
  "- Smoke test UI + API"
) | ForEach-Object { Write-Host $_ }


