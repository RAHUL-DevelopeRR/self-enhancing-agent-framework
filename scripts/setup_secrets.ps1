# PowerShell script to upload API keys from .env as encrypted GitHub Actions secrets via `gh cli`

param (
    [string]$Repo = ""
)

Write-Host "Checking GitHub CLI authentication..." -ForegroundColor Cyan
gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: You are not logged into GitHub CLI." -ForegroundColor Red
    Write-Host "Please run: gh auth login" -ForegroundColor Yellow
    exit 1
}

$envFile = Join-Path $PSScriptRoot "..\.env"
if (-not (Test-Path $envFile)) {
    Write-Host "Error: .env file not found at $envFile. Create it from .env.example first." -ForegroundColor Red
    exit 1
}

$targetRepo = if ($Repo) { "-R $Repo" } else { "" }

Write-Host "Reading secrets from .env and uploading to GitHub Actions..." -ForegroundColor Green

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $val = $parts[1].Trim()
        if ($val) {
            Write-Host "Uploading secret: $key..." -ForegroundColor Cyan
            gh secret set $key --body "$val" $targetRepo
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✔ $key uploaded successfully" -ForegroundColor Green
            }
        }
    }
}

Write-Host "`nAll secrets successfully uploaded to GitHub Actions!" -ForegroundColor Cyan
