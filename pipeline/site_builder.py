import os
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("pipeline.site_builder")

class AutomatedSiteBuilder:
    """Generates modern sample websites for agreed clients and triggers preview deployments."""

    def __init__(self, agent_loop, output_dir: str = "./generated_sites"):
        self.agent_loop = agent_loop
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def build_sample_site(self, client_info: Dict[str, Any]) -> str:
        """Generates a responsive single-page landing site tailored to the client."""
        client_name = client_info.get("title", "Client").replace(" ", "_")
        target_path = self.output_dir / f"{client_name.lower()}_preview.html"

        prompt = (
            f"Generate a modern, responsive HTML5 + Vanilla CSS single-page landing website for:\n"
            f"Business: {client_info.get('title')}\n"
            f"Industry / Niche: {client_info.get('city', 'General')} services\n\n"
            "Include:\n"
            "- Hero section with high-converting headline and call-to-action button\n"
            "- 3 key value propositions / services with modern SVG icons\n"
            "- Testimonials section\n"
            "- Contact / Booking form section\n"
            "- Modern styling: Clean typography, dark or modern vibrant accents, mobile-responsive layout\n"
            "Output ONLY the complete self-contained HTML file content with embedded CSS."
        )

        html_content, _ = await self.agent_loop.execute_task(
            task_type="coding",
            prompt=prompt,
            client_context=client_info
        )

        # Clean code block backticks if present
        cleaned_html = html_content
        if "```html" in cleaned_html:
            cleaned_html = cleaned_html.split("```html")[1].split("```")[0].strip()
        elif "```" in cleaned_html:
            cleaned_html = cleaned_html.split("```")[1].split("```")[0].strip()

        target_path.write_text(cleaned_html, encoding="utf-8")
        logger.info(f"Sample site generated successfully at: {target_path}")
        return str(target_path.absolute())
