import os
import json
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from guardrails import setup_guardrails, safe_completion, GuardrailViolation

load_dotenv()
litellm.suppress_debug_info = True

class QueryPlan(BaseModel):
    query_type: str = Field(
        description="Must be exactly one of: 'global', 'local', 'drift', or 'basic'."
    )
    reasoning: str = Field(
        description="Detailed explanation of why this query type was selected based on the user request."
    )
    target_entities: list[str] = Field(
        description="Key entities mentioned or implied in the query (e.g., ['GridOS', 'Elena Rostova'])."
    )
    keywords: list[str] = Field(
        description="Key terms or keywords extracted from the user query."
    )

class PlanningAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        self.litellm_model = self.model_name
        if "/" not in self.litellm_model:
            self.litellm_model = f"openai/{self.litellm_model}"

        self.active_key = self.openai_key or self.gemini_key
        if not self.active_key:
            print("[Warning] No API keys found in .env. Heuristic fallback will be used.")

        self.client_type = "litellm" if self.active_key else None

        # Register LiteLLM native guardrail hooks (once per process)
        setup_guardrails(
            gemini_key=self.active_key,
            classifier_model=self.litellm_model,
            enable_llm_classifier=True,
        )

    def plan_query(self, user_query: str) -> QueryPlan:
        """
        Analyzes the user query and selects the most appropriate GraphRAG search type.
        """
        system_instruction = (
            "You are an expert GraphRAG Query Planning Agent. Your job is to analyze the user's "
            "question and decide which search type is most appropriate for Microsoft GraphRAG. "
            "Select exactly one of the following query types:\n"
            "1. 'global': Best for high-level summaries, broad thematic questions, and aggregate reports across the entire database "
            "(e.g., 'What are the main goals of the project?', 'Summarize the primary components of this system').\n"
            "2. 'local': Best for specific facts about specific entities (people, software, labs) or direct relationships "
            "(e.g., 'Who is Elena Rostova?', 'What is the relationship between Marcus Vance and Aether Labs?').\n"
            "3. 'drift': Best for exploratory, multi-hop, or open-ended queries that connect multiple entities or trace relationships "
            "indirectly (e.g., 'How did the cyber attack in 2025 impact Elena Rostova's work indirectly?', 'Since Marcus Vance managed the security breach, how does his role link back to Geneva tech partnerships?').\n"
            "4. 'basic': Best for standard keyword/similarity searches looking for specific raw textual facts, exact dates, "
            "or code syntax details that do not require knowledge graph traversal (e.g., 'What programming language is GridOS written in?', 'What is the exact date when the project launched?').\n\n"
            "Analyze carefully, explain your reasoning, extract the target entities and keywords, and return a structured JSON response "
            "matching this schema: {\"query_type\": str, \"reasoning\": str, \"target_entities\": list[str], \"keywords\": list[str]}"
        )

        if not self.client_type:
            return self._heuristic_fallback(user_query)

        try:
            # safe_completion() unwraps BadRequestError into GuardrailViolation
            response = safe_completion(
                model=self.litellm_model,
                messages=[
                    {"role": "system", "content": system_instruction,
                        "cache_control": {
                        "type": "ephemeral",
                        "ttl": "7200s"
                    }
                    },
                    {"role": "user", "content": user_query,
                        "cache_control": {
                        "type": "ephemeral",
                        "ttl": "7200s"
                    }
                    }
                ],
                api_key=self.active_key,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            return QueryPlan(**data)
        except GuardrailViolation:
            raise  # propagate so main.py can show a friendly message
        except Exception as e:
            print(f"[Error] LiteLLM API failed: {e}. Falling back to heuristic planner.")
            return self._heuristic_fallback(user_query)

    def _heuristic_fallback(self, query: str) -> QueryPlan:
        """
        Simple keyword-based heuristic fallback if LLM is unavailable.
        """
        q = query.lower()
        reasoning = "Heuristic fallback: "
        query_type = "basic"
        entities = []
        keywords = []

        if any(w in q for w in ["summary", "overall", "summarize", "main goals", "challenges", "themes", "overview"]):
            query_type = "global"
            reasoning += "Query requests high-level summary/themes across the entire dataset."
        elif any(w in q for w in ["who is", "what is the relationship", "relation", "connect", "between"]):
            query_type = "local"
            reasoning += "Query requests specific facts/relations of designated entities."
        elif any(w in q for w in ["impact", "affect", "since", "consequences", "explore", "how did"]):
            query_type = "drift"
            reasoning += "Query requires exploration of indirect relations and pathway traversal."
        else:
            query_type = "basic"
            reasoning += "Query appears to be a specific lookup of facts or dates."

        for candidate in ["elena rostova", "marcus vance", "gridos", "quantum battery", "aether labs", "project aether", "cyber attack"]:
            if candidate in q:
                entities.append(candidate.title())

        words = q.split()
        keywords = [w for w in words if len(w) > 4 and w not in ["about", "their", "there", "would", "could", "should"]]

        return QueryPlan(
            query_type=query_type,
            reasoning=reasoning,
            target_entities=entities,
            keywords=keywords
        )
