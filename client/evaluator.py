import uuid
import logging
from typing import Dict, Any, Tuple
from .context import ContextEngine
from .memory import EpisodicMemory

logger = logging.getLogger("client.evaluator")

class AgenticLoop:
    """Implements the Draft -> Critic -> Refine loop across heterogeneous free models."""

    def __init__(self, context_engine: ContextEngine, memory: EpisodicMemory, broker_client):
        self.context_engine = context_engine
        self.memory = memory
        self.broker = broker_client
        self.quality_threshold = 0.85
        self.max_refinements = 2

    async def execute_task(self, task_type: str, prompt: str, client_context: Dict[str, Any] = None) -> Tuple[str, float]:
        episode_id = str(uuid.uuid4())
        
        # 1. Draft Phase
        messages = self.context_engine.assemble_context(task_type, prompt, client_context)
        draft_resp = await self.broker.chat(
            messages=messages,
            task=task_type,
            role="draft",
            max_tokens=2048
        )
        current_content = draft_resp.content
        current_model = draft_resp.model
        current_provider = draft_resp.provider

        critic_score = 0.9
        critic_feedback = "Approved on first pass."

        # 2. Refinement Loop (Max 2 iterations)
        for iteration in range(self.max_refinements):
            # Critic evaluates output
            critic_prompt = (
                f"Task Type: {task_type}\n"
                f"Original User Request: {prompt}\n\n"
                f"Candidate Draft:\n{current_content}\n\n"
                "Evaluate this output. Return your evaluation strictly formatted as:\n"
                "SCORE: <float between 0.0 and 1.0>\n"
                "CRITIQUE: <specific actionable flaws, or 'None' if high quality>"
            )
            critic_resp = await self.broker.chat(
                messages=[{"role": "user", "content": critic_prompt}],
                task="critic",
                role="critic",
                max_tokens=512
            )

            # Parse score & critique
            crit_text = critic_resp.content
            score = 0.85
            feedback = ""
            for line in crit_text.splitlines():
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.replace("SCORE:", "").strip())
                    except ValueError:
                        pass
                elif line.startswith("CRITIQUE:"):
                    feedback = line.replace("CRITIQUE:", "").strip()

            critic_score = score
            critic_feedback = feedback or crit_text

            if critic_score >= self.quality_threshold or "none" in critic_feedback.lower():
                logger.info(f"Draft passed evaluation with score {critic_score:.2f}")
                break

            logger.info(f"Refining output (iteration {iteration + 1}, score {critic_score:.2f})...")
            # Refine draft using feedback
            refine_prompt = (
                f"Your previous draft had issues:\n{critic_feedback}\n\n"
                "Please generate an improved, production-ready version resolving all critiques."
            )
            messages.append({"role": "assistant", "content": current_content})
            messages.append({"role": "user", "content": refine_prompt})

            refine_resp = await self.broker.chat(
                messages=messages,
                task=task_type,
                role="refiner",
                max_tokens=2048
            )
            current_content = refine_resp.content

        # Record episode into SQLite memory
        self.memory.record_episode(
            episode_id=episode_id,
            task_type=task_type,
            prompt=prompt,
            draft=draft_resp.content,
            critic_feedback=critic_feedback,
            final_content=current_content,
            model=current_model,
            provider=current_provider,
            score=critic_score,
            success=(critic_score >= 0.75)
        )

        return current_content, critic_score
