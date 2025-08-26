"""
generate_lesson_ollama.py - Script to generate a lesson using Ollama directly
"""

import json
import argparse
import re
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_lesson(subject, topic):
    """
    Generate a lesson using Ollama
    """
    try:
        # Check if Ollama is available and running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                logger.error("Ollama API is not responding")
                return None
        except Exception as e:
            logger.error(f"Failed to connect to Ollama API: {e}")
            return None

        # Define the prompt for generating a lesson
        prompt = f"""
        Create a structured lesson on the topic of {topic} within the subject of {subject} in the context of ancient Indian wisdom.

        The lesson should include:
        1. A title that captures the essence of the lesson
        2. A relevant Sanskrit shloka or verse
        3. An English translation of the shloka
        4. A detailed explanation of the concept, including its significance in ancient Indian knowledge systems
        5. A practical activity for students to engage with the concept
        6. A reflective question for students to ponder

        Format the response as a JSON object with the following structure:
        {{
            "title": "The title of the lesson",
            "shloka": "The Sanskrit shloka or verse",
            "translation": "English translation of the shloka",
            "explanation": "Detailed explanation of the concept",
            "activity": "A practical activity for students",
            "question": "A reflective question for students"
        }}

        Ensure that the content is authentic, respectful of the tradition, and educationally valuable.
        """

        # Use deepseek-r1:1.5b model as it's smaller and more stable
        model = "deepseek-r1:1.5b"
        logger.info(f"Using Ollama model: {model}")

        # Generate the lesson using Ollama API
        logger.info("Generating lesson with Ollama...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": "You are an expert in ancient Indian wisdom traditions and should always respond with valid JSON. Your responses should have title, shloka, translation, explanation, activity, and question fields.",
                "stream": False
            }
        )

        if response.status_code != 200:
            logger.error(f"Ollama API request failed: {response.text}")
            return None

        # Get the output
        response_json = response.json()
        output = response_json.get('response', '').strip()
        logger.info(f"Ollama output received. Length: {len(output)} characters")

        # Try to parse the JSON response
        try:
            # First try direct JSON parsing
            try:
                lesson = json.loads(output)
                logger.info("Successfully parsed JSON directly")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown or text
                json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
                if json_match:
                    lesson_text = json_match.group(1)
                    lesson = json.loads(lesson_text)
                    logger.info("Found and parsed JSON from code block")
                else:
                    # Try to find JSON object directly in text
                    json_match = re.search(r'({.*})', output, re.DOTALL)
                    if json_match:
                        lesson_text = json_match.group(1)
                        lesson = json.loads(lesson_text)
                        logger.info("Found and parsed JSON from text")
                    else:
                        logger.error("Could not find valid JSON in response")
                        return None

            # Validate the lesson structure
            required_fields = ["title", "shloka", "translation", "explanation", "activity", "question"]
            for field in required_fields:
                if field not in lesson:
                    raise ValueError(f"Missing required field: {field}")

            logger.info("Successfully generated and validated lesson")
            return lesson

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw output: {output}")
            return None
        except ValueError as e:
            logger.error(f"Invalid lesson structure: {e}")
            return None

    except Exception as e:
        logger.error(f"Error generating lesson: {e}")
        return None

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Generate a lesson using Ollama")
    parser.add_argument("--subject", required=True, help="Subject of the lesson (e.g., Ved, Ganita, Yoga)")
    parser.add_argument("--topic", required=True, help="Topic of the lesson (e.g., Sound, Mathematics, Meditation)")

    args = parser.parse_args()

    lesson = generate_lesson(args.subject, args.topic)

    if lesson:
        print(json.dumps(lesson, indent=2, ensure_ascii=False))
    else:
        print("Failed to generate lesson")

if __name__ == "__main__":
    main()
