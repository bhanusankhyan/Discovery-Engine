import os
import json
import pandas as pd
import numpy as np
import litellm
import lancedb
import pyarrow as pa
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")

# Read mock data
with open("mock_graph_data.json", "r", encoding="utf-8") as f:
    graph_data = json.load(f)

# Define embedding function
def get_embedding(text: str) -> list[float]:
    print(f"Generating embedding for: '{text[:40]}...'")
    resp = litellm.embedding(
        model="gemini/gemini-embedding-001",
        input=[text],
        api_key=api_key
    )
    return resp.data[0].embedding

# Create output folder
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Text units
chunks = graph_data.get("text_chunks", [])
text_units_data = []
for c in chunks:
    text_units_data.append({
        "id": c["id"],
        "text": c["content"],
        "n_tokens": len(c["content"]) // 4,
        "document_id": c["source"]
    })
text_units_df = pd.DataFrame(text_units_data)
text_units_df.to_parquet(output_dir / "text_units.parquet")
print("Saved text_units.parquet")

# 2. Entities
entities_list = graph_data.get("entities", [])
entities_data = []
for ent in entities_list:
    name = ent["name"]
    text_unit_ids = [c["id"] for c in chunks if name.lower() in c["content"].lower()]
    degree = sum(
        1 for r in graph_data.get("relationships", [])
        if r["source"].lower() == name.lower() or r["target"].lower() == name.lower()
    )
    emb = get_embedding(ent["description"])
    entities_data.append({
        "id": ent["id"],
        "title": name,
        "type": ent["type"],
        "description": ent["description"],
        "human_readable_id": ent["id"],
        "description_embedding": emb,
        "text_unit_ids": text_unit_ids,
        "degree": max(1, degree)
    })
entities_df = pd.DataFrame(entities_data)
entities_df.to_parquet(output_dir / "entities.parquet")
print("Saved entities.parquet")

# 3. Communities
communities_data = [
    {
        "id": "1",
        "community": 1,
        "title": "Grid Technology",
        "level": 0,
        "parent": None,
        "children": [],
        "entity_ids": ["project_aether", "quantum_battery_core", "gridos"],
        "relationship_ids": [],
        "text_unit_ids": ["chunk_3", "chunk_4"]
    },
    {
        "id": "2",
        "community": 2,
        "title": "Aether Leadership",
        "level": 0,
        "parent": None,
        "children": [],
        "entity_ids": ["project_aether", "dr_elena_rostova", "aether_labs"],
        "relationship_ids": [],
        "text_unit_ids": ["chunk_1"]
    },
    {
        "id": "3",
        "community": 3,
        "title": "Security Operations",
        "level": 0,
        "parent": None,
        "children": [],
        "entity_ids": ["marcus_vance", "cyber_attack_2025", "gridos"],
        "relationship_ids": [],
        "text_unit_ids": ["chunk_2"]
    }
]
communities_df = pd.DataFrame(communities_data)
communities_df.to_parquet(output_dir / "communities.parquet")
print("Saved communities.parquet")

# 4. Community Reports
reports_list = graph_data.get("community_reports", [])
reports_data = []
for idx, r in enumerate(reports_list, start=1):
    findings_str = "\n".join([f"- {f}" for f in r["findings"]])
    full_content = r["title"] + "\n\nSummary: " + r["summary"] + "\n\nKey Findings:\n" + findings_str
    emb = get_embedding(full_content)
    reports_data.append({
        "id": str(idx),
        "community": idx,
        "title": r["title"],
        "summary": r["summary"],
        "full_content": full_content,
        "full_content_embedding": emb,
        "rank": 1.0,
        "level": 0
    })
community_reports_df = pd.DataFrame(reports_data)
community_reports_df.to_parquet(output_dir / "community_reports.parquet")
print("Saved community_reports.parquet")

# 5. Relationships
relationships_list = graph_data.get("relationships", [])
relationships_data = []
for idx, rel in enumerate(relationships_list, start=1):
    src = rel["source"]
    tgt = rel["target"]
    text_unit_ids = [
        c["id"] for c in chunks
        if src.lower() in c["content"].lower() or tgt.lower() in c["content"].lower()
    ]
    relationships_data.append({
        "id": f"rel_{idx}",
        "human_readable_id": str(idx),
        "source": src,
        "target": tgt,
        "description": rel["description"],
        "combined_degree": 2,
        "weight": rel["weight"],
        "text_unit_ids": text_unit_ids
    })
relationships_df = pd.DataFrame(relationships_data)
relationships_df.to_parquet(output_dir / "relationships.parquet")
print("Saved relationships.parquet")

# 6. Covariates (Empty placeholder)
covariates_df = pd.DataFrame(columns=[
    "id", "covariate_type", "subject", "object", "type", "status", "source", "description"
])
covariates_df.to_parquet(output_dir / "covariates.parquet")
print("Saved covariates.parquet")

# 7. Write LanceDB vector tables
lancedb_dir = "./output/lancedb"
db = lancedb.connect(lancedb_dir)

# Schema definition for LanceDB table
schema = pa.schema([
    ("id", pa.string()),
    ("vector", pa.list_(pa.float32(), 3072)),
    ("create_date", pa.string()),
    ("update_date", pa.string())
])

# Entity descriptions
for col_name in ["entity_description_embeddings", "entity_description"]:
    print(f"Writing LanceDB collection '{col_name}'...")
    table = db.create_table(col_name, schema=schema, exist_ok=True)
    rows = []
    for ent in entities_data:
        rows.append({
            "id": ent["id"],
            "vector": ent["description_embedding"],
            "create_date": None,
            "update_date": None
        })
    table.add(rows)

# Text units
for col_name in ["text_unit_embeddings", "text_unit_text"]:
    print(f"Writing LanceDB collection '{col_name}'...")
    table = db.create_table(col_name, schema=schema, exist_ok=True)
    rows = []
    for c in text_units_data:
        emb = get_embedding(c["text"])
        rows.append({
            "id": c["id"],
            "vector": emb,
            "create_date": None,
            "update_date": None
        })
    table.add(rows)

# Community full content
for col_name in ["community_full_content_embeddings", "community_full_content"]:
    print(f"Writing LanceDB collection '{col_name}'...")
    table = db.create_table(col_name, schema=schema, exist_ok=True)
    rows = []
    for r in reports_data:
        rows.append({
            "id": r["id"],
            "vector": r["full_content_embedding"],
            "create_date": None,
            "update_date": None
        })
    table.add(rows)

print("\nAll mock Parquet tables and LanceDB vector stores successfully constructed!")
