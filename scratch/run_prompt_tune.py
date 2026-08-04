import subprocess
import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

log_file_path = "prompt_tune_python.log"

try:
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Starting prompt tuning script...\n")
        log_file.write(f"OPENAI_API_KEY is configured in os.environ: {bool(os.environ.get('OPENAI_API_KEY'))}\n")
        log_file.write(f"GRAPHRAG_API_KEY is configured in os.environ: {bool(os.environ.get('GRAPHRAG_API_KEY'))}\n")
        log_file.flush()

        cmd = [
            "python", "-m", "graphrag", "prompt-tune",
            "--root", ".",
            "--domain", "quick commerce in India",
            "--limit", "5"
        ]
        log_file.write(f"Executing command: {' '.join(cmd)}\n")
        log_file.flush()

        # Run process, redirecting output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=os.environ
        )

        log_file.write(f"\n--- STDOUT ---\n")
        log_file.write(result.stdout)
        log_file.write(f"\n--- STDERR ---\n")
        log_file.write(result.stderr)
        log_file.write(f"\nExit Code: {result.returncode}\n")
        log_file.flush()
        
        print(f"Subprocess finished with exit code {result.returncode}.")
        print(f"All outputs written to {log_file_path}.")

except Exception as e:
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- Python Exception ---\n")
        log_file.write(traceback.format_exc())
    print(f"Python script crashed. Traceback written to {log_file_path}.")
