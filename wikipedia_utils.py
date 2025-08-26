"""
wikipedia_utils.py - Utility functions for fetching and processing Wikipedia content
"""

import wikipedia
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("wikipedia.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache directory for Wikipedia content
CACHE_DIR = "wikipedia_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(subject: str, topic: str) -> str:
    """Get the cache file path for a given subject and topic"""
    sanitized_subject = subject.lower().replace(" ", "_")
    sanitized_topic = topic.lower().replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{sanitized_subject}_{sanitized_topic}.json")

def load_from_cache(subject: str, topic: str) -> Optional[Dict[str, Any]]:
    """Load Wikipedia content from cache if available"""
    cache_path = get_cache_path(subject, topic)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if cache is less than 7 days old
                if time.time() - data.get("timestamp", 0) < 7 * 24 * 60 * 60:
                    logger.info(f"Loaded Wikipedia content from cache for {subject}/{topic}")
                    return data
        except Exception as e:
            logger.warning(f"Error loading from cache: {str(e)}")
    return None

def save_to_cache(subject: str, topic: str, data: Dict[str, Any]) -> None:
    """Save Wikipedia content to cache"""
    cache_path = get_cache_path(subject, topic)
    try:
        # Add timestamp for cache expiration
        data["timestamp"] = time.time()
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Wikipedia content to cache for {subject}/{topic}")
    except Exception as e:
        logger.warning(f"Error saving to cache: {str(e)}")

def search_wikipedia(query: str, max_results: int = 5) -> List[str]:
    """Search Wikipedia for relevant articles"""
    try:
        results = wikipedia.search(query, results=max_results)
        logger.info(f"Wikipedia search for '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error searching Wikipedia: {str(e)}")
        return []

def get_wikipedia_content(title: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get content from a Wikipedia article"""
    try:
        # Set language to English
        wikipedia.set_lang("en")
        
        # Get the page
        page = wikipedia.page(title, auto_suggest=True)
        
        # Get summary (introduction)
        summary = page.summary
        
        # Get full content
        content = page.content
        
        # Get URL
        url = page.url
        
        logger.info(f"Successfully fetched Wikipedia content for '{title}'")
        return summary, content, url
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"Disambiguation error for '{title}': {str(e)}")
        # Try the first option if it's a disambiguation page
        if e.options:
            logger.info(f"Trying first disambiguation option: {e.options[0]}")
            return get_wikipedia_content(e.options[0])
        return None, None, None
    except wikipedia.exceptions.PageError as e:
        logger.error(f"Page error for '{title}': {str(e)}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error getting Wikipedia content for '{title}': {str(e)}")
        return None, None, None

def get_relevant_wikipedia_info(subject: str, topic: str) -> Dict[str, Any]:
    """
    Get relevant Wikipedia information for a subject and topic
    
    Args:
        subject: The main subject (e.g., "Ved", "Ganita", "Yoga")
        topic: The specific topic within the subject (e.g., "Sound", "Algebra", "Asana")
        
    Returns:
        Dictionary containing Wikipedia information
    """
    # Check cache first
    cached_data = load_from_cache(subject, topic)
    if cached_data:
        return cached_data
    
    # Prepare search queries
    queries = [
        f"{topic} in {subject}",
        f"{topic} {subject} ancient India",
        f"{topic} in ancient Indian {subject}",
        f"{subject} {topic}"
    ]
    
    # Map subject names to their more common Wikipedia terms
    subject_mapping = {
        "ved": "Vedas",
        "veda": "Vedas",
        "ganita": "Indian mathematics",
        "yoga": "Yoga",
        "ayurveda": "Ayurveda",
        "darshana": "Hindu philosophy",
        "vastu": "Vastu shastra"
    }
    
    # Add mapped subject if available
    mapped_subject = subject_mapping.get(subject.lower())
    if mapped_subject:
        queries.append(f"{topic} in {mapped_subject}")
        queries.append(f"{mapped_subject} {topic}")
    
    # Search for each query and collect results
    all_results = []
    for query in queries:
        results = search_wikipedia(query)
        all_results.extend(results)
    
    # Remove duplicates while preserving order
    unique_results = []
    for result in all_results:
        if result not in unique_results:
            unique_results.append(result)
    
    # Get content for the most relevant result
    if unique_results:
        summary, content, url = get_wikipedia_content(unique_results[0])
        
        if summary and content:
            # Prepare data
            wiki_data = {
                "subject": subject,
                "topic": topic,
                "wikipedia": {
                    "title": unique_results[0],
                    "summary": summary,
                    "content": content,
                    "url": url,
                    "related_articles": unique_results[1:5] if len(unique_results) > 1 else []
                }
            }
            
            # Save to cache
            save_to_cache(subject, topic, wiki_data)
            
            return wiki_data
    
    # Return empty data if nothing found
    return {
        "subject": subject,
        "topic": topic,
        "wikipedia": {
            "title": None,
            "summary": None,
            "content": None,
            "url": None,
            "related_articles": []
        }
    }

if __name__ == "__main__":
    # Test the functions
    test_subject = "Ganita"
    test_topic = "Algebra"
    
    print(f"Testing Wikipedia integration for {test_subject}/{test_topic}")
    wiki_info = get_relevant_wikipedia_info(test_subject, test_topic)
    
    if wiki_info["wikipedia"]["title"]:
        print(f"Found Wikipedia article: {wiki_info['wikipedia']['title']}")
        print(f"URL: {wiki_info['wikipedia']['url']}")
        print(f"Summary: {wiki_info['wikipedia']['summary'][:200]}...")
        print(f"Related articles: {wiki_info['wikipedia']['related_articles']}")
    else:
        print(f"No Wikipedia information found for {test_subject}/{test_topic}")
