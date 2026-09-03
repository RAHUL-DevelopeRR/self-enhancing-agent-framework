from typing import List, Dict, Any, Optional
from .memory import EpisodicMemory

class ContextEngine:
    """Assembles the mutable prompt context dynamically without token bloat."""

    def __init__(self, memory: EpisodicMemory):
        self.memory = memory
        self.core_system_prompt = (
            "You are an autonomous AI Agent operating within an agile B2B development and marketing pipeline. "
            "You write clean, modular code, draft polite and concise outbound messages, and strictly follow "
            "accumulated operational policies and guidelines."
        )

    def assemble_context(self, task_type: str, user_prompt: str, client_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        messages = []

        # 1. Base identity
        messages.append({"role": "system", "content": self.core_system_prompt})

        # 2. Retrieve top lessons from past episodes (zero-token local RAG)
        lessons = self.memory.retrieve_relevant_lessons(query=user_prompt, task_type=task_type, top_k=3)
        if lessons:
            lesson_bullets = "\n".join([f"- [Confidence {l['confidence']:.2f}] {l['lesson']}" for l in lessons])
            policy_prompt = (
                "### LEARNED POLICIES FROM PAST EXECUTION (CRITICAL RULES):\n"
                f"{lesson_bullets}\n"
                "Incorporate these rules into your output to avoid past failure modes."
            )
            messages.append({"role": "system", "content": policy_prompt})

        # 3. Client metadata (if applicable)
        if client_metadata:
            meta_str = "\n".join([f"{k}: {v}" for k, v in client_metadata.items()])
            messages.append({"role": "system", "content": f"### CURRENT CLIENT CONTEXT:\n{meta_str}"})

        # 4. Actual task
        messages.append({"role": "user", "content": user_prompt})
        return messages
