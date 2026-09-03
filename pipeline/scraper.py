import os
import httpx
import logging
from typing import List, Dict, Any

logger = logging.getLogger("pipeline.scraper")

class ApifyLeadScraper:
    """Automates lead generation and contact discovery via Apify Cloud Actors."""

    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("APIFY_TOKEN")
        self.base_url = "https://api.apify.com/v2"

    async def scrape_google_places_leads(self, location: str, search_query: str, max_items: int = 15) -> List[Dict[str, Any]]:
        """Extracts local business leads (phone, website, address) via Apify Google Places Actor."""
        if not self.api_token:
            logger.warning("No APIFY_TOKEN configured; returning synthetic demo leads.")
            return [
                {
                    "title": f"Demo {search_query.capitalize()} Clinic",
                    "website": "https://example-dental-demo.com",
                    "phone": "+1-555-0199",
                    "city": location,
                    "rating": 4.2
                }
            ]

        actor_id = "compass~crawler-google-places"
        run_url = f"{self.base_url}/acts/{actor_id}/run-sync-get-dataset-items?token={self.api_token}"

        payload = {
            "searchStringsArray": [f"{search_query} in {location}"],
            "maxCrawledPlacesPerSearch": max_items,
            "language": "en"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(run_url, json=payload)
                resp.raise_for_status()
                items = resp.json()

                leads = []
                for item in items:
                    website = item.get("website")
                    if website:
                        leads.append({
                            "title": item.get("title", "Business Lead"),
                            "website": website,
                            "phone": item.get("phone", ""),
                            "address": item.get("address", ""),
                            "city": location,
                            "rating": item.get("totalScore", 0.0)
                        })
                logger.info(f"Successfully scraped {len(leads)} leads from Apify.")
                return leads
            except Exception as e:
                logger.error(f"Apify scraping failed: {e}")
                return []
