import sys
import logging
from pathlib import Path

# Reconfigure stdout/stderr to use utf-8 encoding and line-buffering
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

# Set up logging to print directly to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)

# Import GraphRAG index CLI entry point
from graphrag.cli.index import index_cli

def main():
    print(">>> Starting custom line-buffered GraphRAG indexer...", flush=True)
    try:
        index_cli(
            root_dir=Path(r"C:\Users\LENOVO\.gemini\antigravity\scratch\graph_rag_planner"),
            method="standard",
            verbose=True,
            cache=True,
            dry_run=False,
            skip_validation=False
        )
        print(">>> Indexing complete!", flush=True)
    except Exception as e:
        print(f">>> Indexing failed with exception: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
