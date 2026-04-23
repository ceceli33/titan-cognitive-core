"""
AKBASCORE V13.0 — Internet Forager
Autonomous knowledge acquisition from RSS feeds, Wikipedia, and arXiv.

The forager runs on a background schedule, gathering content,
scoring its importance, and feeding it to the learning pipeline.
"""

import requests
import feedparser
import random
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from config.hardware import ForageConfig


@dataclass
class ForagedContent:
    title: str
    summary: str
    source: str
    url: str
    importance: float = 0.3
    raw_text: str = ""

    def full_text(self) -> str:
        return f"{self.title}. {self.summary}"


class InternetForager:
    """
    Autonomous internet forager.
    
    Sources:
    - Major news RSS feeds (BBC, NYT, Google News)
    - arXiv cs.AI + HuggingFace blog (tech papers)
    - Wikipedia random article API
    
    Each piece of content is scored for importance before storage.
    """

    def __init__(self, config: ForageConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AkbasCore-TITAN/13.0 (Educational Research Bot)'
        })

    # ----------------------------------------------------------------
    # FETCH METHODS
    # ----------------------------------------------------------------

    def fetch_news(self, limit_per_feed: int = 5) -> List[ForagedContent]:
        results = []
        for url in self.config.NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit_per_feed]:
                    results.append(ForagedContent(
                        title=entry.get('title', '').strip(),
                        summary=entry.get('summary', '').strip(),
                        source='news',
                        url=entry.get('link', ''),
                    ))
            except Exception as e:
                print(f"  ⚠️  News feed error ({url[:40]}): {e}")
        return results

    def fetch_wikipedia(self) -> Optional[ForagedContent]:
        try:
            r = self.session.get(self.config.WIKIPEDIA_API, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return ForagedContent(
                    title=data.get('title', ''),
                    summary=data.get('extract', '')[:800],
                    source='wikipedia',
                    url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                )
        except Exception as e:
            print(f"  ⚠️  Wikipedia error: {e}")
        return None

    def fetch_tech_papers(self, limit_per_feed: int = 3) -> List[ForagedContent]:
        results = []
        for url in self.config.TECH_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit_per_feed]:
                    results.append(ForagedContent(
                        title=entry.get('title', '').strip(),
                        summary=entry.get('summary', '')[:600].strip(),
                        source='tech',
                        url=entry.get('link', ''),
                    ))
            except Exception as e:
                print(f"  ⚠️  Tech feed error: {e}")
        return results

    # ----------------------------------------------------------------
    # SCORING
    # ----------------------------------------------------------------

    def score_importance(self, content: ForagedContent,
                          user_interests: List[str] = None) -> float:
        """
        Heuristic importance scoring.
        In a production system, replace with embedding cosine similarity
        against a user interest profile vector.
        """
        text = content.full_text().lower()
        score = 0.25  # base

        # Length bonus
        score += min(0.25, len(text) / 2000)

        # Source bonus
        source_bonus = {'tech': 0.15, 'wikipedia': 0.10, 'news': 0.05}
        score += source_bonus.get(content.source, 0.0)

        # Interest matching
        if user_interests:
            matches = sum(1 for i in user_interests if i.lower() in text)
            score += min(0.25, matches * 0.08)

        # Small random jitter (novelty simulation)
        score += random.uniform(0, 0.05)

        return round(min(1.0, score), 3)

    # ----------------------------------------------------------------
    # MAIN FORAGE LOOP
    # ----------------------------------------------------------------

    def forage(self, user_interests: List[str] = None,
               verbose: bool = True) -> List[ForagedContent]:
        """
        Full foraging tour. Returns list sorted by importance (desc).
        """
        if verbose:
            print("\n🌐 Foraging tour started...")

        all_content: List[ForagedContent] = []

        # Gather from all sources
        news = self.fetch_news()
        all_content.extend(news)
        if verbose:
            print(f"   📰 News: {len(news)} articles")

        wiki = self.fetch_wikipedia()
        if wiki:
            all_content.append(wiki)
            if verbose:
                print(f"   📚 Wikipedia: '{wiki.title[:50]}'")

        tech = self.fetch_tech_papers()
        all_content.extend(tech)
        if verbose:
            print(f"   🔬 Tech papers: {len(tech)}")

        # Score each item
        for item in all_content:
            item.importance = self.score_importance(item, user_interests)
            item.raw_text = item.full_text()[:500]

        # Sort by importance
        all_content.sort(key=lambda x: x.importance, reverse=True)

        if verbose:
            print(f"   ✅ Total: {len(all_content)} items | "
                  f"Top: '{all_content[0].title[:45]}...' "
                  f"({all_content[0].importance:.2f})" if all_content else "")

        return all_content
