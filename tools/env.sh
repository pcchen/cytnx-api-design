#!/usr/bin/env bash
# Shared environment for audit tooling. Source or reference PY.
export PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/python"
