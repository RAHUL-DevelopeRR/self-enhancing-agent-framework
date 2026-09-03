import os
import logging
from typing import Dict, Any

logger = logging.getLogger("pipeline.social_showcase")

class SocialShowcaseEngine:
    """Analyzes completed client sites, generates marketing copy & creative prompts, and handles social posting."""

    def __init__(self, agent_loop):
        self.agent_loop = agent_loop

    async def analyze_and_create_post(self, client_info: Dict[str, Any], site_path: str) -> Dict[str, str]:
        """Analyzes client website and produces Instagram caption, hashtags, and image prompt."""
        prompt = (
            f"Analyze this client project and generate a high-performing Instagram Post & Story:\n"
            f"Client Name: {client_info.get('title')}\n"
            f"City: {client_info.get('city')}\n"
            f"Website Path: {site_path}\n\n"
            "Format the response strictly as:\n"
            "IMAGE_PROMPT: <DALL-E / Flux prompt for a sleek 3D mockup of the website on modern smartphone & laptop devices>\n"
            "CAPTION: <engaging Instagram caption highlighting the digital transformation, speed, and modern UI>\n"
            "HASHTAGS: <8-10 relevant hashtags>"
        )

        showcase_plan, _ = await self.agent_loop.execute_task(
            task_type="reasoning",
            prompt=prompt,
            client_context=client_info
        )

        image_prompt, caption, hashtags = "", "", ""
        for line in showcase_plan.splitlines():
            if line.startswith("IMAGE_PROMPT:"):
                image_prompt = line.replace("IMAGE_PROMPT:", "").strip()
            elif line.startswith("CAPTION:"):
                caption = line.replace("CAPTION:", "").strip()
            elif line.startswith("HASHTAGS:"):
                hashtags = line.replace("HASHTAGS:", "").strip()

        return {
            "image_prompt": image_prompt or f"Modern 3D web showcase for {client_info.get('title')}",
            "caption": caption or f"Excited to reveal the modern new digital experience for {client_info.get('title')}! 🚀",
            "hashtags": hashtags or "#WebDesign #ModernUI #DigitalTransformation"
        }
