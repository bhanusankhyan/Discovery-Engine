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
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.litellm_model = f"gemini/{self.model_name}"

        # Load real parquet tables from output dir
        print("[Engine] Loading Parquet artifacts...")
        self.tables = load_parquet_tables(Path("./output"))

        # Build GraphRAG configuration
        print(f"[Engine] Building GraphRAG config with model: {self.litellm_model}")
        self.config = build_graphrag_config(
            api_key=self.gemini_key,
            llm_model=self.litellm_model,
        )

        if self.gemini_key:
            self.client_type = "litellm"
            # Setup LiteLLM native guardrail hooks
            setup_guardrails(
                gemini_key=self.gemini_key,
                classifier_model=self.litellm_model,
                enable_llm_classifier=True,
            )
        else:
            self.client_type = None
            print("[Warning] No GEMINI_API_KEY found.")

    def run_query(self, query: str, plan) -> str:
        """
        Routes the query based on the QueryPlan structure (query_type, target_entities, keywords).
        """
        q_type = plan.query_type.lower()
        loop = asyncio.get_event_loop()

        try:
            if q_type == "global":
                return loop.run_until_complete(run_global_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                ))
            elif q_type == "local":
                return loop.run_until_complete(run_local_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
            elif q_type == "drift":
                return loop.run_until_complete(run_drift_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
            else: # basic
                return loop.run_until_complete(run_basic_search(
                    query=query,
                    config=self.config,
                    tables=self.tables,
                    lancedb_dir="./output/lancedb",
                ))
        except GuardrailViolation as e:
            raise e
        except Exception as e:
            return f"[Search Error] Search execution failed: {e}"

    def global_search(self, query: str, keywords: list[str]) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_global_search(
            query=query,
            config=self.config,
            tables=self.tables,
        ))

    def local_search(self, query: str, target_entities: list[str]) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_local_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))

    def drift_search(self, query: str, target_entities: list[str]) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_drift_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))

    def basic_search(self, query: str, keywords: list[str]) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_basic_search(
            query=query,
            config=self.config,
            tables=self.tables,
            lancedb_dir="./output/lancedb",
        ))
