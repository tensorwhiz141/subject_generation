"""
knowledge_store.py - Module for storing and retrieving generated lessons
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("knowledge_store.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the knowledge store directory
KNOWLEDGE_STORE_DIR = "knowledge_store"
os.makedirs(KNOWLEDGE_STORE_DIR, exist_ok=True)

def get_lesson_path(subject: str, topic: str) -> str:
    """Get the file path for a lesson based on subject and topic"""
    sanitized_subject = subject.lower().replace(" ", "_")
    sanitized_topic = topic.lower().replace(" ", "_")
    return os.path.join(KNOWLEDGE_STORE_DIR, f"{sanitized_subject}_{sanitized_topic}.json")

def save_lesson(lesson_data: Dict[str, Any]) -> bool:
    """
    Save a lesson to the knowledge store (always overwrites existing)

    Args:
        lesson_data: Dictionary containing the lesson data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract subject and topic from the lesson data
        subject = lesson_data.get("subject", "unknown")
        topic = lesson_data.get("topic", "unknown")

        # Check if lesson already exists to determine version
        lesson_path = get_lesson_path(subject, topic)
        version = "1.0"

        if os.path.exists(lesson_path):
            try:
                with open(lesson_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_version = existing_data.get("metadata", {}).get("version", "1.0")
                    # Increment version number
                    version_parts = existing_version.split(".")
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    version = f"{major}.{minor + 1}"
            except Exception:
                version = "1.1"  # Default if we can't read existing version

        # Add metadata with updated version
        lesson_data["metadata"] = {
            "timestamp": time.time(),
            "date_created": datetime.now().isoformat(),
            "version": version,
            "last_updated": datetime.now().isoformat()
        }

        # Save to file (overwrites existing)
        with open(lesson_path, 'w', encoding='utf-8') as f:
            json.dump(lesson_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Lesson on {subject}/{topic} saved to knowledge store (version {version})")
        return True

    except Exception as e:
        logger.error(f"Error saving lesson to knowledge store: {str(e)}")
        return False

def get_lesson(subject: str, topic: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a lesson from the knowledge store

    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson

    Returns:
        Optional[Dict[str, Any]]: The lesson data if found, None otherwise
    """
    try:
        lesson_path = get_lesson_path(subject, topic)
        if os.path.exists(lesson_path):
            with open(lesson_path, 'r', encoding='utf-8') as f:
                lesson_data = json.load(f)
            logger.info(f"Lesson on {subject}/{topic} retrieved from knowledge store")
            return lesson_data
        else:
            logger.info(f"No lesson found for {subject}/{topic} in knowledge store")
            return None

    except Exception as e:
        logger.error(f"Error retrieving lesson from knowledge store: {str(e)}")
        return None

def list_lessons() -> List[Dict[str, str]]:
    """
    List all lessons in the knowledge store

    Returns:
        List[Dict[str, str]]: List of dictionaries with subject and topic
    """
    try:
        lessons = []
        for filename in os.listdir(KNOWLEDGE_STORE_DIR):
            if filename.endswith(".json"):
                try:
                    subject, topic = filename[:-5].split("_", 1)
                    lessons.append({
                        "subject": subject.replace("_", " "),
                        "topic": topic.replace("_", " ")
                    })
                except ValueError:
                    # Skip files that don't follow the naming convention
                    continue

        return lessons

    except Exception as e:
        logger.error(f"Error listing lessons in knowledge store: {str(e)}")
        return []

def search_lessons(query: str) -> List[Dict[str, Any]]:
    """
    Search for lessons in the knowledge store

    Args:
        query: Search query

    Returns:
        List[Dict[str, Any]]: List of matching lessons
    """
    try:
        results = []
        query = query.lower()

        for filename in os.listdir(KNOWLEDGE_STORE_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(KNOWLEDGE_STORE_DIR, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lesson_data = json.load(f)

                    # Check if query matches subject, topic, or content
                    subject = lesson_data.get("subject", "").lower()
                    topic = lesson_data.get("topic", "").lower()
                    title = lesson_data.get("title", "").lower()
                    explanation = lesson_data.get("explanation", "").lower()

                    if (query in subject or query in topic or
                        query in title or query in explanation):
                        results.append(lesson_data)

                except Exception:
                    # Skip files that can't be parsed
                    continue

        return results

    except Exception as e:
        logger.error(f"Error searching lessons in knowledge store: {str(e)}")
        return []

if __name__ == "__main__":
    # Test the module
    test_lesson = {
        "subject": "Veda",
        "topic": "Sound",
        "title": "The Sacred Sound in Vedic Tradition",
        "shloka": "ॐ अग्निमीळे पुरोहितं यज्ञस्य देवम् ऋत्विजम्",
        "translation": "Om, I praise Agni, the priest of the sacrifice, the divine, the ritual performer.",
        "explanation": "In Vedic tradition, sound (shabda) is considered not just a physical phenomenon but a spiritual one.",
        "activity": "Recite the shloka aloud thrice, paying attention to the vibration you feel in different parts of your body.",
        "question": "How does the concept of sound in Vedic tradition differ from the modern scientific understanding of sound?"
    }

    # Save the test lesson
    save_lesson(test_lesson)

    # Retrieve the test lesson
    retrieved_lesson = get_lesson("Veda", "Sound")
    if retrieved_lesson:
        print(f"Retrieved lesson: {retrieved_lesson['title']}")

    # List all lessons
    all_lessons = list_lessons()
    print(f"Total lessons in store: {len(all_lessons)}")
