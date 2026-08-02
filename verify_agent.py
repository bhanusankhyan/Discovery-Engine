import os
import sys
import time
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Import local components
from planner import PlanningAgent
from search_engine import GraphRAGQueryEngine

def get_color_for_type(q_type: str):
    q_type = q_type.lower()
    if q_type == "global":
        return Fore.CYAN
    elif q_type == "local":
        return Fore.YELLOW
    elif q_type == "drift":
        return Fore.MAGENTA
    elif q_type == "basic":
        return Fore.GREEN
    else:
        return Fore.WHITE

def run_tests():
    print(f"{Fore.BLUE}[System] Initializing Planner and Search Engine...")
    planner = PlanningAgent()
    engine = GraphRAGQueryEngine()
    print(f"{Fore.GREEN}[System] Initialization complete! (Client: {getattr(planner, 'client_type', 'None')}, Model: {getattr(planner, 'model_name', 'None')})\n")

    test_queries = [
        ("Global Search Query", "What are the overall themes, objectives, and challenges of the project?"),
        ("Local Search Query", "Who is Dr. Elena Rostova and what did she design?"),
        ("Drift Search Query", "Since there was a cyber attack in 2025, how does that relate to Marcus Vance and GridOS?"),
        ("Basic Search Query", "What was the exact date when the project was launched?")
    ]

    for label, test_q in test_queries:
        print(f"{Fore.CYAN}======================================================================")
        print(f"{Fore.WHITE}{Style.BRIGHT}Testing: {label}")
        print(f"{Fore.WHITE}Query:   {Fore.YELLOW}{test_q}")
        print(f"{Fore.CYAN}======================================================================")
        
        # 1. Run Planner
        plan = planner.plan_query(test_q)
        if not plan:
            print(f"{Fore.RED}[Error] Failed to plan query.")
            continue
            
        color = get_color_for_type(plan.query_type)
        print(f"{Fore.WHITE}Planning Decision:")
        print(f"  - Selected Type:  {color}{plan.query_type.upper()}")
        print(f"  - Reasoning:      {plan.reasoning}")
        print(f"  - Target Entities:{plan.target_entities}")
        print(f"  - Keywords:       {plan.keywords}")
        
        # 2. Run Engine
        print(f"\n{Fore.BLUE}Running search engine...")
        result = engine.run_query(test_q, plan)
        
        print(f"\n{color}{Style.BRIGHT}=== SEARCH RESULT ===")
        print(f"{Fore.WHITE}{result}")
        print(f"{color}{Style.BRIGHT}=====================")
        print("\n" * 2)

if __name__ == "__main__":
    run_tests()
