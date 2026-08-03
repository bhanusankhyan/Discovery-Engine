import os
import sys
import shutil
import subprocess
from dotenv import load_dotenv

# Load credentials
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.yaml")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# Sample text to index
SAMPLE_DOCUMENT_CONTENT = """
Project Aether Charter and Technical Documentation
===================================================
Project Aether was officially launched on October 12, 2024, at Aether Labs in Geneva, Switzerland. Under the leadership of Chief Scientist Dr. Elena Rostova, the project's primary mission is to build a decentralized energy grid that eliminates single points of failure in national power transmission. By utilizing smart contracts on a high-throughput blockchain, the grid allows immediate peer-to-peer energy trading between localized microgrids.

Infrastructure & Quantum Core
------------------------------
A core pillar of the project is the Quantum Battery Core (QBC), developed in collaboration with the Geneva Tech Institute. The QBC achieves a round-trip efficiency of 99.4%, a massive leap compared to standard lithium-ion batteries. In Project Aether, QBCs are deployed at localized neighborhood substations to absorb spikes in solar and wind generation.

GridOS Operating System
-----------------------
GridOS is the proprietary distributed operating system written in Rust to manage the power flow, billing, and optimization of the energy grid. The software features an automated market maker (AMM) module that adjusts electricity pricing dynamically based on local supply and demand. Dr. Elena Rostova designed the load-balancing controller that prevents transformers from overheating during peak charging times.

Security Operations
-------------------
Marcus Vance is the Director of Security at Aether Labs. A former defense contractor, Vance is responsible for defending GridOS from cyber threats. On April 3, 2025, a critical incident occurred: a state-sponsored malware breach targeting the GridOS beta (later attributed to the group 'Lumen Strike') resulted in a temporary microgrid blackout in Geneva. Marcus Vance led the team that isolated the breach and restored grid stability in 4 hours.
"""

def setup_graphrag_workspace():
    print("==========================================================")
    print("    MICROSOFT GRAPHRAG WORKSPACE BUILDER (GEMINI 3.5)")
    print("==========================================================")

    # 1. Check graphrag installation
    try:
        import graphrag
        print("[System] Microsoft GraphRAG library is already installed.")
    except ImportError:
        print("[System] installing 'graphrag' library via pip. This may take a minute...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "graphrag"], check=True)
            print("[System] GraphRAG library installed successfully!")
        except Exception as e:
            print(f"[Error] Failed to install graphrag automatically: {e}")
            print("Please run: pip install graphrag")
            return

    # 2. Run graphrag init command
    print("[System] Initializing GraphRAG workspace directory...")
    try:
        # Run init command via subprocess to generate defaults
        # We temporarily backup the existing .env to prevent overwriting
        backup_env = None
        if os.path.exists(ENV_FILE):
            backup_env = os.path.join(PROJECT_ROOT, ".env.backup")
            shutil.copyfile(ENV_FILE, backup_env)
            print("  - Backed up existing .env to .env.backup")

        subprocess.run([sys.executable, "-m", "graphrag.index", "--init", "--root", PROJECT_ROOT], check=True)

        # Restore backup env
        if backup_env:
            shutil.copyfile(backup_env, ENV_FILE)
            os.remove(backup_env)
            print("  - Restored original .env credentials")

    except Exception as e:
        print(f"[Error] Failed to run GraphRAG initialization: {e}")
        return

    # 3. Create input folder and write sample text file
    print("[System] Preparing input directory and files...")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)

    doc_path = os.path.join(INPUT_DIR, "project_aether_docs.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOCUMENT_CONTENT.strip())
    print(f"  - Created sample source document: input/project_aether_docs.txt")

    # 4. Generate custom settings.yaml configured for Gemini 3.5 Flash Lite via OpenAI compatibility
    print("[System] Generating custom settings.yaml for Google Gemini...")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("[Warning] GEMINI_API_KEY not found in .env. Please configure it.")

    # Write custom yaml configuration
    yaml_content = f"""# GraphRAG configuration file tailored for Google Gemini 3.5 API
encoding_model: cl100k_base
skip_workflows: []

llm:
  api_key: {gemini_key}
  type: openai_chat
  model: gpt-4o-mini
  api_base: https://generativelanguage.googleapis.com/v1beta/openai
  max_tokens: 4000
  temperature: 0.0
  request_timeout: 180.0
  concurrent_requests: 20 # Limit concurrency for rate limits

embeddings:
  async_mode: threaded
  llm:
    api_key: {gemini_key}
    type: openai_embedding
    model: text-embedding-004
    api_base: https://generativelanguage.googleapis.com/v1beta/openai

chunks:
  size: 300
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: "input"
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "output"

entity_extraction:
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event,technology,software]
  max_gleanings: 1

summarize_descriptions:
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  enabled: false
  prompt: "prompts/claim_extraction.txt"

community_reports:
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000
"""

    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print("  - Written settings.yaml with Gemini OpenAI compatibility base URL.")

    # 5. Provide next steps
    print("\n[Success] GraphRAG workspace is fully prepared for Gemini 3.5 Flash Lite!")
    print("==========================================================")
    print("  To build the GraphRAG Knowledge Base index, run:")
    print("  python -m graphrag.index --root .")
    print("==========================================================")

    # Offer to run the index generation
    user_choice = input("\nWould you like to start the index build right now? (y/N): ").strip().lower()
    if user_choice in ["y", "yes"]:
        print("\n[System] Launching GraphRAG indexing pipeline...")
        print("[Warning] Indexing may take a few minutes as it extracts nodes, links, and builds community reports.\n")
        try:
            subprocess.run([sys.executable, "-m", "graphrag.index", "--root", PROJECT_ROOT], check=True)
            print("\n[Success] GraphRAG indexing pipeline completed successfully!")
            print("The output artifacts are saved in the 'output' directory.")
        except Exception as e:
            print(f"\n[Error] Indexing failed: {e}")
            print("Make sure your API key has enough quota and is set correctly.")

if __name__ == "__main__":
    setup_graphrag_workspace()
