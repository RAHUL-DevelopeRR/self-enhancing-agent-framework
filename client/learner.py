import sqlite3
import uuid
import logging
from typing import List, Dict, Any
from .memory import EpisodicMemory

logger = logging.getLogger("client.learner")

class ExperienceHarvester:
    """Mines recorded episodes to discover recurring patterns and formulate persistent learned rules."""

    def __init__(self, memory: EpisodicMemory, broker_client):
        self.memory = memory
        self.broker = broker_client

    async def distill_lessons(self, min_episodes: int = 5):
        """Analyzes recent episodes with lower scores or repeated critiques to extract actionable rules."""
        with sqlite3.connect(self.memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT task_type, prompt, critic_feedback, final_content, critic_score 
            FROM episodes 
            ORDER BY created_at DESC LIMIT 20
            """)
            episodes = cursor.fetchall()

        if len(episodes) < min_episodes:
            logger.info(f"Not enough episodes to harvest lessons ({len(episodes)}/{min_episodes}).")
            return

        # Prepare summary for meta-reflection
        refl_items = []
        for ep in episodes:
            refl_items.append(f"Task: {ep[0]} | Score: {ep[4]:.2f}\nFeedback: {ep[2]}\n")

        meta_prompt = (
            "You are an AI Metacognition Engine. Below are logs of recent agent critiques and execution results:\n\n"
            + "\n".join(refl_items[:10]) +
            "\nIdentify the 1-2 most critical recurring mistakes or success patterns. "
            "Formulate actionable, compact rules for future tasks.\n"
            "Format strictly as:\n"
            "RULE: <concise rule text>\n"
            "TASK_TYPE: <coding | cold_outreach | semantic_reply | social_post>\n"
            "TRIGGER: <keywords that should trigger this rule>"
        )

        resp = await self.broker.chat(
            messages=[{"role": "user", "content": meta_prompt}],
            task="reasoning",
            role="metacognition",
            max_tokens=600
        )

        # Parse generated rules
        rule_text = ""
        task_type = "general"
        trigger = ""

        for line in resp.content.splitlines():
            line = line.strip()
            if line.startswith("RULE:"):
                rule_text = line.replace("RULE:", "").strip()
            elif line.startswith("TASK_TYPE:"):
                task_type = line.replace("TASK_TYPE:", "").strip().lower()
            elif line.startswith("TRIGGER:"):
                trigger = line.replace("TRIGGER:", "").strip()

            if rule_text and trigger:
                lesson_id = str(uuid.uuid4())[:8]
                logger.info(f"Distilled new lesson [{task_type}]: {rule_text}")
                self.memory.add_or_update_lesson(
                    lesson_id=lesson_id,
                    task_type=task_type,
                    trigger_pattern=trigger,
                    lesson_text=rule_text,
                    success=True
                )
                rule_text, trigger = "", ""
