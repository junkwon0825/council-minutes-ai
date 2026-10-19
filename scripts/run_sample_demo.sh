#!/usr/bin/env bash
set -euo pipefail

streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501

