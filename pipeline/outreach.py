import os
import logging
from typing import Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger("pipeline.outreach")

class LeadIntent(str, Enum):
    AGREED_INTERESTED = "agreed_interested"
    REQUEST_MORE_INFO = "request_more_info"
    OBJECTION_PRICE = "objection_price"
    UNSUBSCRIBE = "unsubscribe"
    NEUTRAL = "neutral"

class OmnichannelOutreach:
    """Manages cold pitches, follow-ups, and inbound semantic intent classification."""

    def __init__(self, agent_loop, composio_api_key: str = None):
        self.agent_loop = agent_loop
        self.composio_api_key = composio_api_key or os.getenv("COMPOSIO_API_KEY")

    async def draft_cold_proposal(self, lead: Dict[str, Any], channel: str = "email") -> str:
        """Drafts a high-converting personalized pitch based on the lead's business context."""
        prompt = (
            f"Write a high-converting, personalized cold outreach message for:\n"
            f"Business Name: {lead.get('title')}\n"
            f"Current Website: {lead.get('website')}\n"
            f"City: {lead.get('city')}\n"
            f"Target Channel: {channel.upper()} (Email/WhatsApp/Instagram)\n\n"
            "Requirements:\n"
            "- Under 90 words\n"
            "- Mention one specific improvement for their website (e.g. mobile loading speed or modern booking flow)\n"
            "- Offer to build a free 1-page modern sample mockup with zero obligation\n"
            "- Polite, professional, zero cheesy marketing jargon"
        )
        message, score = await self.agent_loop.execute_task(
            task_type="cold_outreach",
            prompt=prompt,
            client_context=lead
        )
        return message

    async def classify_inbound_reply(self, inbound_message: str) -> Tuple[LeadIntent, str]:
        """Classifies client reply to determine if they agreed, asked for details, or declined."""
        prompt = (
            f"Client reply message:\n\"{inbound_message}\"\n\n"
            "Analyze the client's intent and return strictly in this format:\n"
            "INTENT: <AGREED_INTERESTED | REQUEST_MORE_INFO | OBJECTION_PRICE | UNSUBSCRIBE | NEUTRAL>\n"
            "REASONING: <brief 1-sentence rationale>"
        )
        analysis, _ = await self.agent_loop.execute_task(
            task_type="semantic_reply",
            prompt=prompt
        )

        intent = LeadIntent.NEUTRAL
        for line in analysis.splitlines():
            if line.startswith("INTENT:"):
                val = line.replace("INTENT:", "").strip().upper()
                if "AGREED" in val:
                    intent = LeadIntent.AGREED_INTERESTED
                elif "INFO" in val:
                    intent = LeadIntent.REQUEST_MORE_INFO
                elif "PRICE" in val:
                    intent = LeadIntent.OBJECTION_PRICE
                elif "UNSUBSCRIBE" in val or "NO" in val:
                    intent = LeadIntent.UNSUBSCRIBE

        return intent, analysis
