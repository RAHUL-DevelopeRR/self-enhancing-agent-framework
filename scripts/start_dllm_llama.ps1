param(
    [string]$LlamaServer = "llama-server",
    [int]$Port = 8080
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$preset = Join-Path $projectRoot "configs\llama_models.ini"

if (-not (Get-Command $LlamaServer -ErrorAction SilentlyContinue)) {
    throw "llama-server was not found. Install llama.cpp with: winget install llama.cpp"
}

& $LlamaServer --models-preset $preset --host 127.0.0.1 --port $Port
