import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv

# Load credentials
load_dotenv()

PROJECT_ROOT = r"C:\Users\LENOVO\.gemini\antigravity\scratch\graph_rag_planner"
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
KNOWLEDGE_DATA_DIR = os.path.join(PROJECT_ROOT, "knowledge_data")
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.yaml")

def main():
    # 1. Initialize GraphRAG workspace
    print("[1] Initializing GraphRAG workspace (generates default settings & prompts)...")
    env_file = os.path.join(PROJECT_ROOT, ".env")
    backup_env = os.path.join(PROJECT_ROOT, ".env.backup")

    # Back up env file to prevent GraphRAG init from modifying/deleting keys
    if os.path.exists(env_file):
        shutil.copyfile(env_file, backup_env)
        print("  - Backed up .env file")

    try:
        # Run init with force to generate the latest standard settings.yaml structure
        subprocess.run([
            sys.executable, "-m", "graphrag", "init",
            "--root", PROJECT_ROOT,
            "--force",
            "--model", "gpt-4o-mini", 
            "--embedding", "text-embedding-3-large"
        ], check=True)
        print("  - GraphRAG workspace initialized successfully.")
    except Exception as e:
        print(f"  - Warning during initialization: {e}")

    # Restore .env
    if os.path.exists(backup_env):
        shutil.copyfile(backup_env, env_file)
        os.remove(backup_env)
        print("  - Restored .env file")

    # 2. Setup Input Directory
    print("\n[2] Preparing input folder...")
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
        print("  - Cleared existing input/ folder")
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Copy files from knowledge_data to input
    copied_files = 0
    if os.path.exists(KNOWLEDGE_DATA_DIR):
        for fname in os.listdir(KNOWLEDGE_DATA_DIR):
            if fname.endswith(".txt"):
                src = os.path.join(KNOWLEDGE_DATA_DIR, fname)
                dst = os.path.join(INPUT_DIR, fname)
                shutil.copyfile(src, dst)
                print(f"  - Copied source text: {fname}")
                copied_files += 1
    else:
        print(f"  - Error: knowledge_data directory not found at {KNOWLEDGE_DATA_DIR}")
        return

    if copied_files == 0:
        print("  - Error: No text files found to copy.")
        return

    # 3. Patch the generated settings.yaml to point to Gemini API
    print("\n[3] Patching settings.yaml for Google Gemini API compatibilities...")
    if not os.path.exists(SETTINGS_FILE):
        print(f"  - Error: settings.yaml was not generated at {SETTINGS_FILE}")
        return

    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Define exact replacements for completion and embedding authentication sections
    old_completion = """    auth_method: api_key # or azure_managed_identity
    api_key: ${GRAPHRAG_API_KEY} # set this in the generated .env file, or remove if managed identity"""

    new_completion = """    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
    api_base: https://generativelanguage.googleapis.com/v1beta/openai"""

    content = content.replace(old_completion, new_completion)

    old_embedding = """    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}"""

    new_embedding = """    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
    api_base: https://generativelanguage.googleapis.com/v1beta/openai"""

    content = content.replace(old_embedding, new_embedding)

    # Let's write the patched content back to settings.yaml
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("  - settings.yaml patched and configured successfully!")
    print("\n[Workspace Setup Complete]")

if __name__ == "__main__":
    main()
