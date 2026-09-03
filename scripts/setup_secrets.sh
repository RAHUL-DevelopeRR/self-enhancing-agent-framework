#!/usr/bin/env bash
# Bash script to upload API keys from .env as encrypted GitHub Actions secrets via `gh cli`

set -e

echo "Checking GitHub CLI authentication..."
if ! gh auth status &>/dev/null; then
    echo "Error: Not logged into GitHub CLI. Run 'gh auth login' first."
    exit 1
fi

ENV_FILE="$(dirname "$0")/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found. Create it from .env.example first."
    exit 1
fi

echo "Uploading secrets from .env to GitHub Actions..."

while IFS='=' read -r key value || [ -n "$key" ]; do
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    if [[ -n "$key" && ! "$key" =~ ^# && -n "$value" ]]; then
        echo "Uploading secret: $key..."
        gh secret set "$key" --body "$value"
        echo "✔ $key uploaded successfully"
    fi
done < "$ENV_FILE"

echo "All secrets successfully uploaded to GitHub Actions!"
