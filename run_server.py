"""Server launcher for the PokéDINO FastAPI application.

Can be run directly from the project root:
    python run_server.py
"""

import sys
from pathlib import Path
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=str(SRC_DIR),
    )
