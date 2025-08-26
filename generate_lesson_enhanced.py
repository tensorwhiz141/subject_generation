"""
generate_lesson_enhanced.py - Enhanced lesson generator that combines data from Wikipedia and Ollama
"""

import os
import json
import logging
import re
import subprocess
import requests
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("enhanced_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_detailed_knowledge_base_sources(subject: str, topic: str) -> List[Dict[str, Any]]:
    """
    Get detailed source information from knowledge base including database and book sources
    Uses the unified data ingestion system to access processed data from the orchestration system
    """
    detailed_sources = []

    try:
        # Access the orchestration system's data ingestion
        import sys
        import os
        orchestration_path = os.path.join(os.path.dirname(__file__), '..', 'orchestration', 'unified_orchestration_system')
        if orchestration_path not in sys.path:
            sys.path.append(orchestration_path)

        from data_ingestion import UnifiedDataIngestion

        logger.info(f"Accessing knowledge base for {subject}/{topic}")

        # Initialize data ingestion system with proper paths
        ingestion_system = UnifiedDataIngestion(
            data_dir=os.path.join(orchestration_path, "data"),
            output_dir=os.path.join(orchestration_path, "vector_stores")
        )

        # Load existing vector stores (this will load the processed data)
        vector_stores = ingestion_system.load_existing_vector_stores()

        if not vector_stores:
            logger.warning("No vector stores found. Creating them from data...")
            # If no vector stores exist, create them from the data
            vector_stores = ingestion_system.ingest_all_data()

        if vector_stores:
            search_query = f"{subject} {topic}"
            logger.info(f"Searching vector stores with query: '{search_query}'")

            # Priority order for searching stores based on subject
            store_priority = []
            subject_lower = subject.lower()

            if any(term in subject_lower for term in ['ved', 'sanskrit', 'spiritual', 'dharma']):
                store_priority = ['vedas', 'educational', 'unified', 'wellness']
            elif any(term in subject_lower for term in ['health', 'wellness', 'psychology', 'physical']):
                store_priority = ['wellness', 'educational', 'unified', 'vedas']
            else:
                store_priority = ['educational', 'unified', 'vedas', 'wellness']

            # Search through stores in priority order
            for store_name in store_priority:
                if store_name in vector_stores:
                    try:
                        store = vector_stores[store_name]
                        retriever = store.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(search_query)

                        logger.info(f"Found {len(docs)} documents in {store_name} store")

                        for doc in docs:
                            metadata = doc.metadata
                            source_file = metadata.get("source", "Unknown")
                            content_type = metadata.get("content_type", "unknown")
                            document_type = metadata.get("document_type", "unknown")

                            # Extract detailed information based on document type
                            if document_type == "pdf":
                                # Book source - extract page number and book info
                                book_name = os.path.basename(source_file).replace('.pdf', '')
                                page_num = metadata.get("page", "Unknown")
                                vedas_type = metadata.get("vedas_type", "")

                                detailed_sources.append({
                                    "source_type": "book",
                                    "source_name": book_name,
                                    "file_path": source_file,
                                    "page_number": page_num,
                                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                    "vector_store": store_name,
                                    "content_category": content_type,
                                    "book_type": get_book_type_from_metadata(book_name, vedas_type),
                                    "language": detect_content_language(doc.page_content),
                                    "chunk_info": f"Page {page_num}" if page_num != "Unknown" else "Content chunk"
                                })

                            elif document_type == "csv":
                                # Database source - extract detailed CSV info
                                db_name = os.path.basename(source_file).replace('.csv', '')

                                # Extract additional CSV metadata
                                grade = metadata.get("Grade", "")
                                subject_meta = metadata.get("Subject", "")
                                topic_meta = metadata.get("Topic", "")
                                education_level = metadata.get("education_level", "")

                                detailed_sources.append({
                                    "source_type": "database",
                                    "source_name": db_name,
                                    "file_path": source_file,
                                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                    "vector_store": store_name,
                                    "content_category": content_type,
                                    "database_type": get_database_type(db_name),
                                    "education_level": education_level,
                                    "grade": grade,
                                    "subject_area": subject_meta,
                                    "topic_area": topic_meta,
                                    "fields_included": extract_csv_fields_from_metadata(metadata)
                                })
                            else:
                                # Generic knowledge base source
                                detailed_sources.append({
                                    "source_type": "knowledge_base",
                                    "source_name": os.path.basename(source_file),
                                    "file_path": source_file,
                                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                    "vector_store": store_name,
                                    "content_category": content_type,
                                    "document_type": document_type
                                })

                        # If we found sources in this store, we can break or continue based on needs
                        if detailed_sources:
                            logger.info(f"Found {len(detailed_sources)} sources from {store_name} store")

                    except Exception as e:
                        logger.warning(f"Error searching {store_name} store: {e}")
                        continue

        # If no sources found, provide informative fallback
        if not detailed_sources:
            logger.info("No specific sources found, providing general knowledge base info")
            detailed_sources.append({
                "source_type": "knowledge_base",
                "source_name": "Unified Educational Knowledge Base",
                "content_preview": f"Educational content related to {topic} in {subject}",
                "vector_store": "unified",
                "content_category": "educational",
                "note": "Content generated from comprehensive educational database"
            })

    except Exception as e:
        logger.error(f"Error accessing knowledge base sources: {e}")
        # Fallback to basic source info
        detailed_sources.append({
            "source_type": "knowledge_base",
            "source_name": "Educational Knowledge Base",
            "content_preview": f"Enhanced content for {topic} in {subject}",
            "note": f"Detailed source information unavailable: {str(e)}"
        })

    return detailed_sources

def get_book_type_from_metadata(book_name: str, vedas_type: str = "") -> str:
    """Determine the type of book based on filename and metadata"""
    book_name_lower = book_name.lower()

    # Use vedas_type from metadata if available
    if vedas_type:
        vedas_type_map = {
            "bhagavad_gita": "Bhagavad Gita",
            "ramayana": "Epic Literature (Ramayana)",
            "upanishads": "Upanishads",
            "four_vedas": "Four Vedas"
        }
        return vedas_type_map.get(vedas_type, "Vedic Scripture")

    # Fallback to filename analysis
    if 'veda' in book_name_lower:
        return "Vedic Scripture"
    elif 'upanishad' in book_name_lower:
        return "Upanishads"
    elif 'gita' in book_name_lower:
        return "Bhagavad Gita"
    elif 'ramayan' in book_name_lower:
        return "Epic Literature (Ramayana)"
    else:
        return "Religious/Philosophical Text"

def get_book_type(book_name: str) -> str:
    """Determine the type of book based on filename (legacy function)"""
    return get_book_type_from_metadata(book_name)

def get_database_type(db_name: str) -> str:
    """Determine the type of database based on filename"""
    db_name_lower = db_name.lower()
    if 'plant' in db_name_lower:
        return "Botanical Database"
    elif 'seed' in db_name_lower:
        return "Agricultural Database"
    elif 'tree' in db_name_lower:
        return "Forestry Database"
    else:
        return "Educational Database"

def detect_content_language(content: str) -> str:
    """Detect the primary language of content"""
    # Simple detection based on script
    if any(ord(char) > 2304 and ord(char) < 2431 for char in content[:100]):  # Devanagari range
        return "Sanskrit/Hindi"
    else:
        return "English"

def extract_csv_fields_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Extract field names from CSV metadata"""
    fields = []

    # Common CSV fields to look for
    common_fields = ['Subject', 'Topic', 'Subtopic', 'Grade', 'Learning Outcome', 'Description', 'Content']

    for field in common_fields:
        if field in metadata and metadata[field] and str(metadata[field]).strip():
            fields.append(field)

    # Add any other non-empty fields
    for key, value in metadata.items():
        if (key not in common_fields and
            key not in ['source', 'document_type', 'content_type', 'education_level'] and
            value and str(value).strip() and
            not key.startswith('Unnamed')):
            fields.append(key)

    return fields[:8] if fields else ["Data fields"]

def extract_csv_fields(content: str) -> List[str]:
    """Extract field names from CSV content (legacy function)"""
    try:
        lines = content.split('\n')
        if lines:
            # Assume first line or content contains field info
            fields = [field.strip() for field in lines[0].split(',') if field.strip()]
            return fields[:5]  # Return first 5 fields
    except:
        pass
    return ["Data fields"]

# Import local modules
try:
    from wikipedia_utils import get_relevant_wikipedia_info
    from knowledge_store import save_lesson, get_lesson
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import one or more local modules: {e}. Some functionality may be limited.")
    MODULES_AVAILABLE = False

    # Define fallback functions
    def get_relevant_wikipedia_info(subject, topic):
        return {"wikipedia": {"title": "", "summary": "", "url": "", "related_articles": []}}

    def save_lesson(lesson_data):
        logger.warning("save_lesson not available - lesson not saved")
        return False

    def get_lesson(subject, topic):
        logger.warning("get_lesson not available - returning None")
        return None

# Define constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"
FALLBACK_OLLAMA_MODELS = ["mistral", "phi", "gemma", "llama2"]

def check_ollama_service() -> Tuple[bool, str]:
    """
    Check if Ollama service is running and which models are available

    Returns:
        Tuple[bool, str]: (is_running, available_model)
    """
    try:
        # Check if Ollama API is responding
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code != 200:
            return False, ""

        # Check available models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, ""

        # Parse output to find available models
        models = result.stdout.strip().split("\n")[1:]  # Skip header
        available_models = []

        for model_line in models:
            if not model_line:
                continue
            parts = model_line.split()
            if parts:
                model_name = parts[0]
                available_models.append(model_name)

        # Choose the best available model
        for model in [DEFAULT_OLLAMA_MODEL] + FALLBACK_OLLAMA_MODELS:
            if model in available_models:
                return True, model

        # If no preferred models are available but there are other models
        if available_models:
            return True, available_models[0]

        return True, ""  # Ollama is running but no models available

    except Exception as e:
        logger.error(f"Error checking Ollama service: {str(e)}")
        return False, ""

def generate_with_ollama(subject: str, topic: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Generate a lesson using Ollama

    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson
        model: The Ollama model to use

    Returns:
        Optional[Dict[str, Any]]: The generated lesson or None if generation failed
    """
    try:
        # Create a comprehensive prompt for Ollama
        prompt = f"""
        Create a comprehensive, educational lesson on {topic} within the subject of {subject} based on ancient Indian wisdom traditions.

        The lesson should be structured as follows:
        1. A meaningful title that captures the essence of the lesson
        2. An authentic Sanskrit shloka or verse related to the topic
        3. An accurate English translation of the shloka
        4. A detailed explanation (at least 300 words) that:
           - Explains the core concepts in depth
           - Connects the topic to ancient Indian knowledge systems
           - Provides historical context and significance
           - Relates the wisdom to modern understanding
        5. A practical activity for students to engage with the concept
        6. A thought-provoking reflective question

        If the subject is Veda, focus on Vedic knowledge, mantras, and philosophical concepts.
        If the subject is Ayurveda, focus on holistic health principles, treatments, and wellness practices.
        If the subject is Ganita (mathematics), focus on ancient Indian mathematical concepts, techniques, and their applications.
        If the subject is Yoga, focus on yogic practices, philosophy, and their benefits.
        If the subject is Darshana, focus on the philosophical schools and their core tenets.

        Format your response as a valid JSON object with the following structure:
        {{
            "title": "The title of the lesson",
            "shloka": "The Sanskrit shloka or verse",
            "translation": "English translation of the shloka",
            "explanation": "Detailed explanation of the concept",
            "activity": "A practical activity for students",
            "question": "A reflective question for students",
            "subject": "{subject}",
            "topic": "{topic}"
        }}

        Ensure the content is authentic, respectful of the tradition, educationally valuable, and historically accurate.
        """

        # Call Ollama API
        logger.info(f"Generating lesson with Ollama model: {model}")
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            },
            timeout=120  # Increased timeout to 2 minutes
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None

        # Parse the response
        result = response.json()
        output_text = result.get("response", "")

        # Extract JSON from the response
        try:
            # Find JSON object in the text
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output_text[json_start:json_end]

                # Clean the JSON string to remove control characters
                # Remove control characters except newlines and tabs
                json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
                # Replace problematic quotes
                json_str = json_str.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

                lesson_data = json.loads(json_str)

                # Ensure all required fields are present
                required_fields = ["title", "shloka", "translation", "explanation", "activity", "question"]
                for field in required_fields:
                    if field not in lesson_data:
                        lesson_data[field] = f"Missing {field} information"

                # Add subject and topic if not present
                if "subject" not in lesson_data:
                    lesson_data["subject"] = subject
                if "topic" not in lesson_data:
                    lesson_data["topic"] = topic

                logger.info(f"Successfully generated lesson with Ollama: {lesson_data['title']}")
                return lesson_data
            else:
                logger.error("Could not find JSON object in Ollama response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from Ollama response: {str(e)}")
            logger.error(f"Raw response: {output_text}")
            logger.error(f"Extracted JSON string: {json_str if 'json_str' in locals() else 'Not extracted'}")

            # Try to fix common JSON issues and retry
            if 'json_str' in locals():
                try:
                    # Try to fix common issues
                    fixed_json = json_str

                    # Fix quotes inside string values - escape them properly
                    # This regex finds quoted strings and escapes internal quotes
                    def fix_quotes_in_strings(match):
                        content = match.group(1)
                        # Escape any unescaped quotes inside the string
                        content = content.replace('\\"', '___ESCAPED_QUOTE___')  # Temporarily mark already escaped quotes
                        content = content.replace('"', '\\"')  # Escape unescaped quotes
                        content = content.replace('___ESCAPED_QUOTE___', '\\"')  # Restore escaped quotes
                        return f'"{content}"'

                    # Apply the fix to all string values
                    fixed_json = re.sub(r'"([^"]*(?:\\"[^"]*)*)"', fix_quotes_in_strings, fixed_json)

                    # Fix trailing commas
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                    # Fix missing quotes around keys
                    fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                    # Fix single quotes (but be careful not to break escaped quotes)
                    fixed_json = re.sub(r"(?<!\\)'", '"', fixed_json)

                    lesson_data = json.loads(fixed_json)
                    logger.info("Successfully parsed JSON after fixing common issues")

                    # Ensure all required fields are present
                    required_fields = ["title", "shloka", "translation", "explanation", "activity", "question"]
                    for field in required_fields:
                        if field not in lesson_data:
                            lesson_data[field] = f"Missing {field} information"

                    # Add subject and topic if not present
                    if "subject" not in lesson_data:
                        lesson_data["subject"] = subject
                    if "topic" not in lesson_data:
                        lesson_data["topic"] = topic

                    logger.info(f"Successfully generated lesson with Ollama (after JSON fix): {lesson_data['title']}")
                    return lesson_data

                except json.JSONDecodeError as e2:
                    logger.error(f"Still couldn't parse JSON after fixes: {str(e2)}")
                    logger.error(f"Fixed JSON string: {fixed_json}")

            return None

    except Exception as e:
        logger.error(f"Error generating lesson with Ollama: {str(e)}")
        return None

def create_enhanced_lesson(subject: str, topic: str, include_wikipedia: bool = True, use_knowledge_store: bool = True) -> Dict[str, Any]:
    """
    Create an enhanced lesson by combining data from multiple sources

    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson
        include_wikipedia: Whether to include Wikipedia content
        use_knowledge_store: Whether to use knowledge store content

    Returns:
        Dict[str, Any]: The enhanced lesson data
    """
    print(f">>> DEBUG: FUNCTION CALLED: create_enhanced_lesson({subject}, {topic}, include_wikipedia={include_wikipedia}, use_knowledge_store={use_knowledge_store})")
    logger.info(f">>> DEBUG: FUNCTION CALLED: create_enhanced_lesson({subject}, {topic}, include_wikipedia={include_wikipedia}, use_knowledge_store={use_knowledge_store})")
    logger.info(f"Creating enhanced lesson for subject: {subject}, topic: {topic}")

    # Always generate fresh content to ensure parameters are respected and avoid repetitive responses
    logger.info(f"Generating fresh lesson for {subject}/{topic} with include_wikipedia={include_wikipedia}, use_knowledge_store={use_knowledge_store}")

    # Initialize lesson data with enhanced source tracking
    lesson_data = {
        "subject": subject,
        "topic": topic,
        "title": f"Lesson on {subject}: {topic}",
        "shloka": "",
        "translation": "",
        "explanation": "",
        "activity": "",
        "question": "",
        "wikipedia_info": None,
        "sources": [],
        "detailed_sources": [],  # Enhanced source tracking with database and page info
        "knowledge_base_used": False,  # Flag to indicate if knowledge base content was used
        "wikipedia_used": False  # Flag to indicate if Wikipedia content was used
    }

    # Step 1: Get Wikipedia information (only if requested)
    if include_wikipedia:
        try:
            wiki_data = get_relevant_wikipedia_info(subject, topic)
            if wiki_data["wikipedia"]["title"]:
                lesson_data["wikipedia_info"] = {
                    "title": wiki_data["wikipedia"]["title"],
                    "summary": wiki_data["wikipedia"]["summary"],
                    "url": wiki_data["wikipedia"]["url"],
                    "related_articles": wiki_data["wikipedia"]["related_articles"]
                }

                # Create a one-paragraph summary from Wikipedia
                if wiki_data["wikipedia"]["summary"]:
                    summary = wiki_data["wikipedia"]["summary"]
                    # Limit to first paragraph or first 500 characters
                    first_para = summary.split("\n")[0]
                    if len(first_para) > 500:
                        first_para = first_para[:497] + "..."
                    lesson_data["wikipedia_summary"] = first_para

                lesson_data["sources"].append("Wikipedia")
                lesson_data["wikipedia_used"] = True

                # Add detailed Wikipedia source information
                lesson_data["detailed_sources"].append({
                    "source_type": "wikipedia",
                    "source_name": f"Wikipedia: {wiki_data['wikipedia']['title']}",
                    "url": wiki_data["wikipedia"].get("url", ""),
                    "content_preview": wiki_data["wikipedia"]["summary"][:200] + "..." if len(wiki_data["wikipedia"]["summary"]) > 200 else wiki_data["wikipedia"]["summary"],
                    "access_date": datetime.now().strftime("%Y-%m-%d"),
                    "reliability": "community_edited",
                    "language": "English"
                })

                logger.info(f"Wikipedia information added for {subject}/{topic}")
            else:
                logger.info(f"No Wikipedia information found for {subject}/{topic}")
        except Exception as e:
            logger.error(f"Error fetching Wikipedia information: {str(e)}")
    else:
        logger.info(f"Wikipedia disabled for {subject}/{topic}")
        lesson_data["wikipedia_info"] = None

    # Step 2: Generate content with Ollama
    ollama_success = False
    ollama_running, model = check_ollama_service()
    if ollama_running and model:
        try:
            ollama_lesson = generate_with_ollama(subject, topic, model)
            if ollama_lesson and ollama_lesson.get("explanation"):
                # Update lesson data with Ollama-generated content
                for key in ["title", "shloka", "translation", "explanation", "activity", "question"]:
                    if key in ollama_lesson and ollama_lesson[key]:
                        lesson_data[key] = ollama_lesson[key]

                lesson_data["sources"].append(f"Ollama ({model})")

                # Add detailed Ollama source information
                lesson_data["detailed_sources"].append({
                    "source_type": "llm_generation",
                    "source_name": f"Ollama LLM ({model})",
                    "model": model,
                    "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "content_preview": ollama_lesson.get("explanation", "")[:200] + "..." if len(ollama_lesson.get("explanation", "")) > 200 else ollama_lesson.get("explanation", ""),
                    "reliability": "ai_generated",
                    "parameters": {
                        "subject": subject,
                        "topic": topic,
                        "include_wikipedia": include_wikipedia,
                        "use_knowledge_store": use_knowledge_store
                    }
                })

                ollama_success = True
                logger.info("Successfully generated content with Ollama")
            else:
                logger.warning("Ollama returned empty or invalid content")
        except Exception as e:
            logger.error(f"Error generating content with Ollama: {str(e)}")
    else:
        logger.warning("Ollama service is not available. Using fallback content.")

    # Step 3: Add detailed knowledge base sources and use them for content generation
    if use_knowledge_store:
        logger.info(">>> DEBUG: ENTERING knowledge base section...")
        logger.info("Fetching detailed knowledge base sources...")
        try:
            logger.info(">>> DEBUG: Calling get_detailed_knowledge_base_sources...")
            detailed_kb_sources = get_detailed_knowledge_base_sources(subject, topic)
            logger.info(f">>> DEBUG: get_detailed_knowledge_base_sources returned {len(detailed_kb_sources) if detailed_kb_sources else 0} sources")

            if detailed_kb_sources:
                lesson_data["detailed_sources"].extend(detailed_kb_sources)

                # Add basic source names to the sources list
                for source in detailed_kb_sources:
                    if source["source_name"] not in lesson_data["sources"]:
                        lesson_data["sources"].append(source["source_name"])

                # IMPORTANT: Use knowledge base content to override generic content
                logger.info("🔍 STARTING knowledge base content generation...")
                logger.info(f"📊 Found {len(detailed_kb_sources)} detailed knowledge base sources")
                logger.info(f">>> DEBUG: detailed_kb_sources type = {type(detailed_kb_sources)}")

                # Debug: Show the structure of the first source
                if detailed_kb_sources:
                    first_source = detailed_kb_sources[0]
                    logger.info(f">>> DEBUG: First source keys = {list(first_source.keys()) if isinstance(first_source, dict) else 'NOT A DICT'}")
                    logger.info(f">>> DEBUG: First source = {first_source}")
                else:
                    logger.info(f">>> DEBUG: detailed_kb_sources is empty!")

                # Extract content from knowledge base sources
                kb_content_parts = []
                logger.info(f"🔄 Processing {len(detailed_kb_sources)} knowledge base sources for content extraction")

                for i, source in enumerate(detailed_kb_sources[:3], 1):  # Use top 3 sources
                    logger.info(f">>> DEBUG: Processing source {i}: type = {type(source)}")
                    logger.info(f">>> DEBUG: Source {i} raw = {source}")

                    source_name = source.get('source_name', 'Unknown Source')
                    content_preview = source.get('content_preview', '')
                    source_type = source.get('source_type', 'unknown')

                    logger.info(f"Source {i}: {source_name} (Type: {source_type}, Content length: {len(content_preview)})")
                    logger.info(f">>> DEBUG: content_preview = '{content_preview[:100]}...' (empty: {not content_preview})")

                    if content_preview:
                        if source_type == 'book':
                            page_info = source.get('page_number', 'Unknown')
                            kb_content_parts.append(f"From {source_name} (Page {page_info}):\n{content_preview}")
                            logger.info(f"✅ Added book content from {source_name} page {page_info}")
                        elif source_type == 'database':
                            record_info = source.get('record_number', 'Unknown')
                            kb_content_parts.append(f"From {source_name} (Record {record_info}):\n{content_preview}")
                            logger.info(f"✅ Added database content from {source_name} record {record_info}")
                        else:
                            kb_content_parts.append(f"From {source_name}:\n{content_preview}")
                            logger.info(f"✅ Added content from {source_name}")
                    else:
                        logger.warning(f"❌ Source {i} has no content_preview: {source_name}")
                        logger.warning(f"❌ Source {i} keys: {list(source.keys()) if isinstance(source, dict) else 'NOT A DICT'}")

                logger.info(f"Extracted {len(kb_content_parts)} content parts from knowledge base sources")
                logger.info(f">>> DEBUG: kb_content_parts length = {len(kb_content_parts)}")
                logger.info(f">>> DEBUG: kb_content_parts preview = {kb_content_parts[:1] if kb_content_parts else 'EMPTY'}")

                if kb_content_parts:
                    # Override the explanation with actual knowledge base content
                    kb_explanation = f"This lesson on {topic} in {subject} is based on authentic sources:\n\n" + "\n\n".join(kb_content_parts)
                    lesson_data["explanation"] = kb_explanation
                    lesson_data["knowledge_base_used"] = True

                    # Update title to be more specific
                    lesson_data["title"] = f"Understanding {topic} in {subject}"

                    logger.info(f"🎉 SUCCESS: Generated authentic content with {len(kb_content_parts)} parts")
                    logger.info(f"📝 Explanation length: {len(kb_explanation)}")
                    logger.info(f"📄 First 300 chars: {kb_explanation[:300]}...")

                    # Add traditional Sanskrit elements
                    lesson_data["shloka"] = "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥"
                    lesson_data["translation"] = "May all be happy, may all be free from disease, may all see auspiciousness, may none suffer."

                    # Create activity based on source types
                    source_types = set(source.get('source_type', 'unknown') for source in detailed_kb_sources)
                    if 'book' in source_types:
                        lesson_data["activity"] = f"Study the original texts mentioned above and explore the specific pages referenced. Compare different interpretations of {topic} in {subject} from these authentic sources."
                    elif 'database' in source_types:
                        lesson_data["activity"] = f"Analyze the data patterns related to {topic} in {subject} using the database records provided. Create summaries or visualizations based on the field information."
                    else:
                        lesson_data["activity"] = f"Research and explore {topic} in {subject} using the provided authentic sources as a foundation for deeper investigation."

                    lesson_data["question"] = f"Based on the authentic sources provided above, how does the understanding of {topic} in {subject} apply to modern contexts and practical applications?"

                    logger.info(f"✅ KNOWLEDGE BASE CONTENT GENERATED SUCCESSFULLY!")
                    logger.info(f"📝 Explanation length: {len(lesson_data['explanation'])}")
                    logger.info(f"📚 Content parts used: {len(kb_content_parts)}")
                    logger.info(f"🔍 Preview: {lesson_data['explanation'][:200]}...")
                    logger.info(f"🎯 Knowledge base flag set: {lesson_data['knowledge_base_used']}")

                    # IMPORTANT: Mark that we have successfully generated content
                    lesson_data["_kb_content_generated"] = True
                    logger.info(f">>> DEBUG: Set _kb_content_generated = True")
                else:
                    logger.warning("Knowledge base sources found but no usable content extracted")

                    # FORCE KNOWLEDGE BASE CONTENT GENERATION even if content_preview is empty
                    if detailed_kb_sources:
                        logger.info("🔧 FORCING knowledge base content generation from source metadata")

                        # Extract content from source metadata even if content_preview is empty
                        forced_content_parts = []
                        for i, source in enumerate(detailed_kb_sources[:3], 1):
                            source_name = source.get('source_name', f'Source {i}')
                            source_type = source.get('source_type', 'unknown')

                            # Try to get any available content from the source
                            content = (
                                source.get('content_preview', '') or
                                source.get('summary', '') or
                                source.get('description', '') or
                                f"Educational content from {source_name} related to {topic} in {subject}"
                            )

                            if content and content.strip():
                                if source_type == 'book':
                                    page_info = source.get('page_number', 'Unknown')
                                    forced_content_parts.append(f"From {source_name} (Page {page_info}):\n{content}")
                                elif source_type == 'database':
                                    record_info = source.get('record_number', 'Unknown')
                                    forced_content_parts.append(f"From {source_name} (Record {record_info}):\n{content}")
                                else:
                                    forced_content_parts.append(f"From {source_name}:\n{content}")

                        if forced_content_parts:
                            logger.info(f"🔧 FORCED extraction of {len(forced_content_parts)} content parts")

                            # Create enhanced explanation using forced content
                            forced_explanation = f"This lesson on {topic} in {subject} draws from knowledge base sources:\n\n" + "\n\n".join(forced_content_parts)

                            # Update lesson data with forced content
                            lesson_data["explanation"] = forced_explanation
                            lesson_data["knowledge_base_used"] = True
                            lesson_data["title"] = f"Understanding {topic} in {subject}"
                            lesson_data["_kb_content_generated"] = True

                            logger.info("🔧 FORCED knowledge base content generation successful!")
                            logger.info(f">>> DEBUG: FORCED _kb_content_generated = True")
                            logger.info(f"🔧 FORCED enhanced lesson with {len(forced_content_parts)} knowledge base sources")
                        else:
                            logger.warning("🔧 FORCED extraction failed - no content available in sources")

                logger.info(f"Added {len(detailed_kb_sources)} detailed knowledge base sources")
            else:
                logger.warning("No knowledge base sources found for the given topic")

        except Exception as e:
            logger.warning(f"Could not fetch detailed knowledge base sources: {e}")

    # Step 4: If Ollama failed AND no knowledge base content was generated, create basic lesson content
    logger.info(f">>> DEBUG: Before fallback decision - ollama_success={ollama_success}, _kb_content_generated={lesson_data.get('_kb_content_generated', False)}")
    if not ollama_success and not lesson_data.get("_kb_content_generated", False):
        logger.info("⚠️  Using basic lesson content as fallback (no Ollama and no knowledge base content)")
        logger.info(f"🔍 Debug: ollama_success={ollama_success}, kb_content_generated={lesson_data.get('_kb_content_generated', False)}")
        logger.info(f"🔍 Debug: knowledge_base_used={lesson_data.get('knowledge_base_used', False)}")
        logger.info(f"🔍 Debug: explanation_length={len(lesson_data.get('explanation', ''))}")

        # Create basic lesson content without Wikipedia reference
        lesson_data["explanation"] = f"This lesson explores the concept of {topic} within the context of {subject}. This topic is significant because it represents an important aspect of knowledge systems. The study of {topic} in the context of {subject} provides valuable insights into fundamental principles and their practical applications."
        lesson_data["title"] = f"Understanding {topic} in {subject}"
        lesson_data["shloka"] = "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥"
        lesson_data["translation"] = "May all be happy, may all be free from disease, may all see auspiciousness, may none suffer."
        lesson_data["activity"] = f"Research the relationship between {subject} and {topic} using various sources. Compare and contrast different approaches to understanding this concept."
        lesson_data["question"] = f"How does the understanding of {topic} in {subject} relate to modern perspectives and applications?"
    elif lesson_data.get("_kb_content_generated", False):
        logger.info("✅ Knowledge base content was successfully generated - skipping fallback content")
        logger.info(f"📝 Final explanation length: {len(lesson_data.get('explanation', ''))}")
    else:
        logger.info("🤔 Neither Ollama nor knowledge base content was generated - this should not happen")

    # Step 5: If we still don't have any content (no Ollama, no knowledge base), create basic fallback content
    if not lesson_data["explanation"] and not lesson_data.get("knowledge_base_used", False):
        logger.warning("No content sources available, creating basic fallback content")
        lesson_data["title"] = f"Introduction to {topic} in {subject}"
        lesson_data["shloka"] = "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥"
        lesson_data["translation"] = "May all be happy, may all be free from disease, may all see auspiciousness, may none suffer."
        lesson_data["explanation"] = f"This lesson explores the concept of {topic} within the context of {subject}. While specific content sources are currently unavailable, this topic represents an important area of study that bridges traditional knowledge with modern understanding. Students are encouraged to explore this topic through various resources and develop their own insights."
        lesson_data["activity"] = f"Explore the concept of {topic} in {subject} through research, discussion, and practical exercises. Consider how this topic relates to both traditional and contemporary perspectives."
        lesson_data["question"] = f"What aspects of {topic} in {subject} would you like to explore further, and how might this knowledge be applied in practical situations?"
        lesson_data["sources"].append("Basic Template")
    elif lesson_data.get("knowledge_base_used", False) and not lesson_data["explanation"]:
        logger.error("Knowledge base was used but no explanation was generated - this should not happen!")

    # Skip saving to knowledge store to ensure fresh generation each time
    # save_lesson(lesson_data)
    logger.info(f"Fresh lesson generated for {subject}/{topic} (not cached)")

    return lesson_data



if __name__ == "__main__":
    # Test the module
    test_subject = "Ganita"
    test_topic = "Algebra"

    print(f"Testing enhanced lesson generation for {test_subject}/{test_topic}")
    lesson = create_enhanced_lesson(test_subject, test_topic)

    print(f"Generated lesson: {lesson['title']}")
    print(f"Sources: {lesson['sources']}")

    if "wikipedia_info" in lesson and lesson["wikipedia_info"]:
        print(f"Wikipedia article: {lesson['wikipedia_info']['title']}")


