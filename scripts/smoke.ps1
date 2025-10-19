param(
  [string]$BaseUrl = "http://localhost:8010"
)

Write-Host "Probing $BaseUrl..." -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/health" | Format-Table
Invoke-RestMethod "$BaseUrl/version" | Format-Table

$edges = Join-Path $PSScriptRoot "..\data\edges.tsv"
$texts = Join-Path $PSScriptRoot "..\data\node_texts.jsonl"
New-Item -Force -ItemType Directory (Split-Path $edges) | Out-Null
"Alice`tcoauthored_with`tCarol" | Out-File -Encoding utf8 $edges
'{"id":"Alice","text":"Author"}' | Out-File -Encoding utf8 $texts

Invoke-RestMethod -Method Post -Uri "$BaseUrl/reload" `
  -ContentType "application/json" -Body (@{edges=(Resolve-Path $edges).Path; texts=(Resolve-Path $texts).Path} | ConvertTo-Json) `
  | Format-Table

Invoke-RestMethod -Method Post -Uri "$BaseUrl/ask?dry_run=true" `
  -ContentType "application/json" -Body (@{question="Who co-authored with Carol?"; topk_paths=3; max_hops=2; neighbor_expand=1; use_rerank=$false; model="ollama/llama3"} | ConvertTo-Json) `
  | Format-List
