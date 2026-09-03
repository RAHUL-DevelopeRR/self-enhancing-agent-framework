import asyncio
import os
import argparse
import logging
from broker.server import router, quota_tracker
from broker.models import ChatRequest, ChatMessage, TaskType
from client.memory import EpisodicMemory
from client.context import ContextEngine
from client.evaluator import AgenticLoop
from client.learner import ExperienceHarvester
from pipeline.scraper import ApifyLeadScraper
from pipeline.outreach import OmnichannelOutreach, LeadIntent
from pipeline.site_builder import AutomatedSiteBuilder
from pipeline.social_showcase import SocialShowcaseEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

class DirectBrokerClient:
    """Invokes the in-process model broker directly without requiring network roundtrips."""
    def __init__(self, router):
        self.router = router

    async def chat(self, messages, task="general", role=None, max_tokens=2048):
        req = ChatRequest(
            messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            task=TaskType(task) if task in TaskType._value2member_map_ else TaskType.GENERAL,
            role=role,
            max_tokens=max_tokens
        )
        return await self.router.route_and_execute(req)

async def run_pipeline(dry_run: bool = False):
    logger.info("Initializing Self-Enhancing Agentic Framework...")

    # 1. Initialize Memory, Context Engine & Loop
    memory = EpisodicMemory("agent_memory.db")
    context_engine = ContextEngine(memory)
    broker_client = DirectBrokerClient(router)
    agent_loop = AgenticLoop(context_engine, memory, broker_client)
    harvester = ExperienceHarvester(memory, broker_client)

    # 2. Pipeline Modules
    scraper = ApifyLeadScraper()
    outreach = OmnichannelOutreach(agent_loop)
    site_builder = AutomatedSiteBuilder(agent_loop)
    showcase = SocialShowcaseEngine(agent_loop)

    logger.info("Step 1: Scraping Leads via Apify...")
    leads = await scraper.scrape_google_places_leads(location="Austin, TX", search_query="Dental Clinic", max_items=2)
    if not leads:
        logger.info("Using simulated lead for demonstration.")
        leads = [{
            "title": "Austin Smiles Dentistry",
            "website": "https://austin-smiles-demo.com",
            "phone": "+1-512-555-0144",
            "city": "Austin, TX"
        }]

    for lead in leads:
        logger.info(f"\nProcessing Lead: {lead['title']} ({lead['website']})")
        
        # Step 2: Draft personalized outreach (Email / WhatsApp)
        logger.info("Step 2: Drafting personalized pitch with Draft->Critic loop...")
        pitch = await outreach.draft_cold_proposal(lead, channel="email")
        print("\n--- GENERATED PROPOSAL ---")
        print(pitch)
        print("--------------------------\n")

        # Step 3: Simulate Client Reply & Semantic Intent Classification
        simulated_reply = "Hey! Loved your idea about our mobile booking. Can you show us a quick sample website?"
        logger.info(f"Step 3: Client replied: '{simulated_reply}'")
        intent, analysis = await outreach.classify_inbound_reply(simulated_reply)
        logger.info(f"Classified Intent: {intent.value}")

        if intent == LeadIntent.AGREED_INTERESTED:
            # Step 4: Build Sample Website
            logger.info("Step 4: Client agreed! Generating responsive sample website...")
            site_path = await site_builder.build_sample_site(lead)
            logger.info(f"Sample website generated: {site_path}")

            # Step 5: Social Media Showcase Engine
            logger.info("Step 5: Generating social showcase & marketing banners...")
            post_data = await showcase.analyze_and_create_post(lead, site_path)
            print("\n--- SOCIAL SHOWCASE PLAN ---")
            print(f"Caption: {post_data['caption']}")
            print(f"Image Prompt: {post_data['image_prompt']}")
            print(f"Hashtags: {post_data['hashtags']}")
            print("----------------------------\n")

    # Step 6: Harvest and crystallize learned lessons into permanent memory
    logger.info("Step 6: Distilling experience into learned lessons (Self-Enhancement)...")
    await harvester.distill_lessons(min_episodes=1)
    logger.info("Agent pipeline iteration complete. Framework successfully self-enhanced.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic verification")
    args = parser.parse_args()
    asyncio.run(run_pipeline(dry_run=args.dry_run))
