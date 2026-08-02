import os
import shutil

PROJECT_ROOT = r"C:\Users\LENOVO\.gemini\antigravity\scratch\graph_rag_planner"
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.yaml")
TEMP_SETTINGS = os.path.join(PROJECT_ROOT, "temp_init", "settings.yaml")

def main():
    if not os.path.exists(TEMP_SETTINGS):
        print("Error: temp settings not found!")
        return
        
    with open(TEMP_SETTINGS, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace keys and add api_base for completion
    old_completion = """    auth_method: api_key # or azure_managed_identity
    api_key: ${GRAPHRAG_API_KEY} # set this in the generated .env file, or remove if managed identity"""
    
    new_completion = """    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
    api_base: https://generativelanguage.googleapis.com/v1beta/openai"""
    
    content = content.replace(old_completion, new_completion)
    
    # Replace keys and add api_base for embedding
    old_embedding = """    auth_method: api_key
    api_key: ${GRAPHRAG_API_KEY}"""
    
    new_embedding = """    auth_method: api_key
    api_key: ${GEMINI_API_KEY}
    api_base: https://generativelanguage.googleapis.com/v1beta/openai"""
    
    content = content.replace(old_embedding, new_embedding)
    
    # Save to PROJECT_ROOT settings.yaml
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Success: settings.yaml configured!")

if __name__ == "__main__":
    main()
