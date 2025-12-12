import logging
import random
from typing import Dict, List, Any

import requests


logger = logging.getLogger("youtube-recommendations")


class YouTubeRecommendationService:
    """Provides YouTube video recommendations for wellness topics.

    Two methods:
    - get_recommendations: Offline/mock generator for basic queries.
    - get_recommendations_with_youtube_api: Uses YouTube Data API v3.
    """

    @staticmethod
    def _build_queries(symptom: str) -> List[str]:
        term = symptom.lower().strip()
        base = [
            f"{term} relief at home",
            f"{term} exercises",
            f"diet for {term}",
            f"foods to avoid for {term}",
            f"breathing techniques for {term}",
            f"yoga for {term}",
        ]
        # Ensure uniqueness and reasonable ordering
        seen = set()
        queries: List[str] = []
        for q in base:
            if q not in seen:
                queries.append(q)
                seen.add(q)
        return queries

    @staticmethod
    def get_recommendations(symptom: str, max_videos: int = 4) -> Dict[str, Any]:
        """Return mock recommendations (no external API).

        Produces stable queries and a small curated set of wellness-oriented channels
        without external calls. Useful as a fallback when API fails or key missing.
        """
        queries = YouTubeRecommendationService._build_queries(symptom)
        seed_channels = [
            {"channel": "Mayo Clinic", "url": "https://www.youtube.com/@MayoClinic"},
            {
                "channel": "Cleveland Clinic",
                "url": "https://www.youtube.com/@ClevelandClinic",
            },
            {
                "channel": "Harvard Health Publishing",
                "url": "https://www.youtube.com/@HarvardHealthPublishing",
            },
            {
                "channel": "Yoga With Adriene",
                "url": "https://www.youtube.com/@yogawithadriene",
            },
            {"channel": "NHS", "url": "https://www.youtube.com/@NHSChoices"},
        ]

        # Shuffle so results look fresh while remaining deterministic enough
        random.shuffle(seed_channels)
        videos = [
            {
                "title": f"Explore: {symptom} — trusted guidance",
                "channelTitle": c["channel"],
                "url": c["url"],
                "thumbnail": None,
                "videoId": None,
            }
            for c in seed_channels[: max(1, min(max_videos, 4))]
        ]

        return {
            "success": True,
            "symptom": symptom,
            "queries": queries,
            "videos": videos,
        }

    @staticmethod
    def get_recommendations_with_youtube_api(
        symptom: str, api_key: str, max_videos: int = 4
    ) -> Dict[str, Any]:
        """Query the YouTube Data API v3 for wellness videos.

        - Uses Search endpoint with safe, health-oriented queries
        - Returns basic metadata suitable for frontend rendering
        """
        queries = YouTubeRecommendationService._build_queries(symptom)
        collected: List[Dict[str, Any]] = []

        try:
            for q in queries:
                if len(collected) >= max_videos:
                    break

                params = {
                    "part": "snippet",
                    "q": q,
                    "type": "video",
                    "maxResults": max(5, max_videos),
                    "key": api_key,
                    "safeSearch": "strict",
                }
                resp = requests.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params=params,
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "YouTube API search failed: %s %s", resp.status_code, resp.text
                    )
                    continue

                data = resp.json()
                items = data.get("items", [])
                for item in items:
                    if len(collected) >= max_videos:
                        break
                    id_obj = item.get("id", {})
                    vid = id_obj.get("videoId")
                    snip = item.get("snippet", {})
                    if not vid or not snip:
                        continue
                    thumbs = snip.get("thumbnails", {})
                    thumb_obj = thumbs.get("medium") or thumbs.get("default") or {}
                    thumbnail_url = thumb_obj.get("url")
                    video = {
                        "title": snip.get("title"),
                        "channelTitle": snip.get("channelTitle"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "thumbnail": thumbnail_url,
                        "videoId": vid,
                    }
                    collected.append(video)

            return {
                "success": True if collected else False,
                "symptom": symptom,
                "queries": queries,
                "videos": collected,
            }

        except Exception as e:
            logger.exception("YouTube API error: %s", e)
            return {
                "success": False,
                "symptom": symptom,
                "queries": queries,
                "videos": [],
                "error": str(e),
            }


"""
YouTube Recommendations Service

Provides YouTube video recommendations for health symptoms and wellness topics.
Uses YouTube Data API v3 to fetch real videos with metadata and thumbnails.
"""

import requests
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YouTubeRecommendationService:
    """Service to fetch YouTube recommendations for health symptoms and wellness topics."""

    # Mapping of symptoms/conditions to relevant YouTube search queries
    SYMPTOM_TO_QUERIES = {
        "fever": [
            "how to reduce fever naturally",
            "fever relief remedies",
            "fever management at home",
        ],
        "cough": [
            "home remedies for cough",
            "dry cough treatment",
            "cough relief exercises",
        ],
        "headache": [
            "migraine relief exercises",
            "tension headache remedies",
            "headache treatment at home",
        ],
        "cold": [
            "common cold natural remedies",
            "flu vs cold treatment",
            "cold recovery tips",
        ],
        "cold_flu": [
            "flu symptoms and treatment",
            "influenza recovery",
            "flu prevention tips",
        ],
        "blood_pressure": [
            "high blood pressure management",
            "hypertension natural remedies",
            "blood pressure control diet",
        ],
        "sugar": [
            "diabetes management",
            "blood sugar control diet",
            "diabetes prevention tips",
        ],
        "chest_pain": [
            "chest pain causes and treatment",
            "when to see doctor for chest pain",
            "heart health prevention",
        ],
        "joint_pain": [
            "arthritis relief exercises",
            "joint pain management",
            "arthritis diet and nutrition",
        ],
        "anxiety": [
            "anxiety relief techniques",
            "breathing exercises for anxiety",
            "meditation for anxiety",
        ],
        "sleep": [
            "improve sleep quality naturally",
            "insomnia treatment",
            "sleep hygiene tips",
        ],
        "digestion": [
            "digestive health tips",
            "acid reflux home remedies",
            "constipation relief",
        ],
        "fitness": [
            "home workout routines",
            "beginner fitness exercises",
            "daily workout motivation",
        ],
        "diet": [
            "healthy eating habits",
            "balanced diet nutrition",
            "weight management diet",
        ],
        "yoga": [
            "beginner yoga for flexibility",
            "yoga for stress relief",
            "morning yoga routine",
        ],
        "meditation": [
            "guided meditation for beginners",
            "daily meditation practice",
            "mindfulness meditation",
        ],
    }

    @staticmethod
    def get_search_queries_for_symptom(symptom: str) -> List[str]:
        """
        Get YouTube search queries for a given symptom.

        Args:
            symptom: User symptom or health concern

        Returns:
            List of search queries relevant to the symptom
        """
        symptom_lower = symptom.lower()

        # Exact match
        if symptom_lower in YouTubeRecommendationService.SYMPTOM_TO_QUERIES:
            return YouTubeRecommendationService.SYMPTOM_TO_QUERIES[symptom_lower]

        # Fuzzy match - find related keywords
        queries = []
        for key, values in YouTubeRecommendationService.SYMPTOM_TO_QUERIES.items():
            if any(word in symptom_lower for word in key.split("_")):
                queries.extend(values)

        # If no match found, create generic queries
        if not queries:
            queries = [
                f"{symptom} treatment and relief",
                f"{symptom} home remedies",
                f"{symptom} management tips",
            ]

        return queries[:3]  # Return top 3 queries

    @staticmethod
    def get_recommendations(symptom: str, max_videos: int = 5) -> Dict[str, any]:
        """
        Get YouTube video recommendations for a symptom.

        This uses a mock/local approach since we don't require an API key.

        Args:
            symptom: User symptom or health concern
            max_videos: Maximum number of videos to return

        Returns:
            Dictionary with search queries and mock video recommendations
        """
        queries = YouTubeRecommendationService.get_search_queries_for_symptom(symptom)

        recommendations = {
            "symptom": symptom,
            "queries": queries,
            "videos": YouTubeRecommendationService._get_mock_videos(
                queries, max_videos
            ),
        }

        return recommendations

    @staticmethod
    def _get_mock_videos(queries: List[str], max_videos: int) -> List[Dict[str, str]]:
        """
        Generate mock YouTube video recommendations based on queries.

        Args:
            queries: List of search queries
            max_videos: Maximum number of videos to return

        Returns:
            List of video recommendation objects with title, url, etc.
        """
        mock_videos = []

        for query in queries:
            # Create a YouTube search URL that users can click
            youtube_search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

            # Align shape with frontend expectations (include 'url', 'title', 'thumbnail', 'description')
            mock_videos.append(
                {
                    "title": f"{query}",
                    "query": query,
                    "url": youtube_search_url,
                    "thumbnail": "https://via.placeholder.com/480x360?text=YouTube+Search",
                    "description": f"Click to explore videos for: {query}",
                }
            )

        return mock_videos[:max_videos]

    @staticmethod
    def get_recommendations_with_youtube_api(
        symptom: str, api_key: str, max_videos: int = 5
    ) -> Dict[str, any]:
        """
        Get YouTube recommendations using YouTube Data API v3.

        Fetches actual videos from YouTube with real metadata and thumbnails.

        Args:
            symptom: User symptom or health concern
            api_key: YouTube Data API v3 key
            max_videos: Maximum number of videos to return

        Returns:
            Dictionary with video recommendations from YouTube
        """
        queries = YouTubeRecommendationService.get_search_queries_for_symptom(symptom)
        all_videos = []

        try:
            for query in queries:
                url = "https://www.googleapis.com/youtube/v3/search"
                # Use minimal, widely supported parameters to reduce 403s
                params = {
                    "part": "snippet",
                    "type": "video",
                    "q": query,
                    "maxResults": 5,
                    "key": api_key,
                    # Prefer embeddable videos
                    "videoEmbeddable": "true",
                    # Hint for localization; optional
                    "regionCode": "US",
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                if "error" in data:
                    logger.error(f"YouTube API error: {data['error']}")
                    return {
                        "symptom": symptom,
                        "queries": queries,
                        "videos": [],
                        "success": False,
                        "error": f"YouTube API error: {data['error']['message']}",
                    }

                if "items" in data:
                    for item in data["items"]:
                        try:
                            thumbs = item["snippet"].get("thumbnails", {})
                            thumb_url = (
                                thumbs.get("high", {}).get("url")
                                or thumbs.get("medium", {}).get("url")
                                or thumbs.get("default", {}).get("url")
                            )

                            video = {
                                "title": item["snippet"].get("title", ""),
                                "videoId": item["id"].get("videoId"),
                                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                                "thumbnail": thumb_url,
                                "description": item["snippet"].get("description", ""),
                                "channelTitle": item["snippet"].get("channelTitle", ""),
                                "publishedAt": item["snippet"].get("publishedAt", ""),
                            }
                            all_videos.append(video)

                            if len(all_videos) >= max_videos:
                                break
                        except KeyError as e:
                            logger.warning(f"Missing field in video data: {e}")
                            continue

                if len(all_videos) >= max_videos:
                    break

            return {
                "symptom": symptom,
                "queries": queries,
                "videos": all_videos[:max_videos],
                "success": True,
                "count": len(all_videos),
            }

        except requests.exceptions.Timeout:
            logger.error("YouTube API request timeout")
            return {
                "symptom": symptom,
                "queries": queries,
                "videos": [],
                "success": False,
                "error": "Request timeout. Please try again.",
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"YouTube API request failed: {str(e)}")
            return {
                "symptom": symptom,
                "queries": queries,
                "videos": [],
                "success": False,
                "error": f"Failed to fetch videos: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Unexpected error in YouTube recommendations: {str(e)}")
            return {
                "symptom": symptom,
                "queries": queries,
                "videos": [],
                "success": False,
                "error": str(e),
            }
