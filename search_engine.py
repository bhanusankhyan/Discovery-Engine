import os
import json
import asyncio
import litellm
from dotenv import load_dotenv
from guardrails import setup_guardrails, GuardrailViolation
from search_workflow import (
    load_parquet_tables,
    build_graphrag_config,
    run_global_search,
    run_local_search,
    run_drift_search,
    run_basic_search,
)
from pathlib import Path

load_dotenv()
litellm.suppress_debug_info = True

class GraphRAGQueryEngine:
    def __init__(self, data_path: str = "mock_graph_data.json"):
        self.data_path = data_path
        
        # Initialize key and model name from env
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        self.litellm_model = self.model_name
        if "/" not in self.litellm_model:
            self.litellm_model = f"openai/{self.litellm_model}"

        # Load real parquet tables from output dir
        print("[Engine] Loading Parquet artifacts...")
        self.tables = load_parquet_tables(Path("./output"))

        # Build GraphRAG configuration
        print(f"[Engine] Building GraphRAG config with model: {self.litellm_model}")
        self.config = build_graphrag_config(
            llm_api_key=self.openai_key or self.gemini_key,
            emb_api_key=self.gemini_key or self.openai_key,
            llm_model=self.litellm_model,
        )

        active_key = self.openai_key or self.gemini_key
        if active_key:
            self.client_type = "litellm"
            # Setup LiteLLM native guardrail hooks
            setup_guardrails(
                gemini_key=active_key,
                classifier_model=self.litellm_model,
                enable_llm_classifier=True,
            )
        else:
            self.client_type = None
            print("[Warning] No API keys found in environment.")

    def run_query(self, query: str, plan) -> str:
        """
        Routes the query based on the QueryPlan structure (query_type, target_entities, keywords).
        """
        q_type = plan.query_type.lower()
        loop = asyncio.get_event_loop()

        try:
            if q_type == "global":
                res = loop.run_until_complete(run_global_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                ))
            elif q_type == "local":
                res = loop.run_until_complete(run_local_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
            elif q_type == "drift":
                res = loop.run_until_complete(run_drift_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
            else: # basic
                res = loop.run_until_complete(run_basic_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
            return self.strip_citations(res)
        except GuardrailViolation as e:
            raise e
        except Exception as e:
            return f"[Search Error] Search execution failed: {e}"

    def strip_citations(self, text: str) -> str:
        """
        Removes GraphRAG source citations in the format [Data: Reports (x, y)],
        strips references to data sources/names (e.g. Report X, Project Aether),
        and cleans up any formatting.
        """
        if not text:
            return text
        import re
        
        # 1. Remove [Data: ...] citation blocks (including +more variations)
        text = re.sub(r"(?i)\[data:.*?\]", "", text)
        
        # 2. Strip file names or dataset/source labels (case-insensitive)
        text = re.sub(r"(?i)\bIndia Q-Commerce Market Analysis(\.txt)?\b", "", text)
        text = re.sub(r"(?i)\bProject Aether\b", "", text)
        text = re.sub(r"(?i)\bGeneva Labs\b", "", text)
        
        # 3. Strip parenthetical report references like (Report 15) or (Reports 12, 16)
        text = re.sub(r"(?i)\(\s*reports?\s*\d+(?:\s*,\s*\d+)*\s*\)", "", text)
        
        # 4. Strip introductory phrases or text references to reports or data sources
        text = re.sub(r"(?i)\b(according to|based on)\s+(the\s+)?(community\s+)?reports?\s*(indicate|show|suggest)?\b", "", text)
        text = re.sub(r"(?i)\b(according to|based on)\s+data\s*(indicate|show|suggest)?\b", "", text)
        text = re.sub(r"(?i)\breports?\s*\d+(?:\s*(?:and|,|to)\s*\d+)*\s*(?:indicate|show|suggest)?\b", "", text)
        
        # 5. Clean up spaces before punctuation marks like periods, commas, etc.
        text = re.sub(r"\s+([.,;!?])", r"\1", text)
        # Clean up double spaces
        text = re.sub(r" {2,}", " ", text)
        
        return text.strip()

    def global_search(self, query: str, keywords: list[str]) -> str:
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(run_global_search(
            query=query,
            config=self.config,
            tables=self.tables,
        ))
        return self.strip_citations(res)

    def local_search(self, query: str, target_entities: list[str]) -> str:
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(run_local_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))
        return self.strip_citations(res)

    def drift_search(self, query: str, target_entities: list[str]) -> str:
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(run_drift_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))
        return self.strip_citations(res)

    def basic_search(self, query: str, keywords: list[str]) -> str:
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(run_basic_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))
        return self.strip_citations(res)
