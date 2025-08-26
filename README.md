# Gurukul AI-Lesson Generator

A dynamic lesson generator that takes a subject and topic, retrieves knowledge from Vedic and modern sources, and produces structured lessons using LangChain pipelines.

## Project Overview

The Gurukul AI-Lesson Generator is designed to create authentic, modular lessons based on ancient Indian wisdom texts and modern knowledge. The system uses a retrieval-augmented generation approach to produce structured JSON lessons.

## Core Features

- **Knowledge Base**: Loads and processes ancient texts (Rigveda, Isha Upanishad, Yoga Sutra, Lilavati, Arthashastra)
- **Vector Storage**: Embeds and stores text chunks in ChromaDB for efficient retrieval
- **Dynamic Lesson Generation**: Creates structured lessons based on subject and topic inputs
- **API Access**: Provides FastAPI endpoints for lesson generation
- **Ollama Integration**: Optional support for local LLMs using Ollama

## Project Structure

```
.
├── texts/                  # Directory containing source texts
│   ├── rigveda.txt
│   ├── isha_upanishad.txt
│   ├── yoga_sutra.txt
│   ├── lilavati.txt
│   └── arthashastra.txt
├── knowledge_store/        # ChromaDB vector database
├── load_vedas.py           # Script to load and process texts
├── generate_lesson.py      # Script to generate lessons
├── ollama_integration.py   # Optional Ollama integration
├── app.py                  # FastAPI application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   CHROMA_PERSIST_DIRECTORY=knowledge_store
   ```

## Usage

### 1. Load the Knowledge Base

Process the text files and create the vector database:

```
python load_vedas.py
```

### 2. Generate a Lesson

Generate a lesson using the command line:

```
python generate_lesson.py --subject "Ved" --topic "Sound"
```

### 3. Run the API Server

Start the FastAPI server:

```
python app.py
```

Access the API at http://localhost:8000

### API Endpoints

- `GET /generate_lesson?subject=Ved&topic=Sound`: Generate a lesson with the specified subject and topic
- `POST /generate_lesson`: Generate a lesson with a JSON request body

## Lesson Structure

Lessons are returned in the following JSON format:

```json
{
  "title": "Lesson 1: The Sound of the Vedas",
  "shloka": "ॐ अग्निमीळे पुरोहितं यज्ञस्य देवम् ऋत्विजम्",
  "translation": "Om, I praise Agni, the priest of the sacrifice, the divine, the ritual performer.",
  "explanation": "In this verse, Agni is invoked as a bridge between humans and devas...",
  "activity": "Recite the shloka aloud thrice. Reflect on the sound and vibration.",
  "question": "What is the role of Agni in a Yajna?"
}
```

## Ollama Integration

For local LLM support, you can use Ollama:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama2`
3. Use the Ollama integration:
   ```python
   from ollama_integration import get_llm
   
   llm = get_llm(use_ollama=True)
   ```

## License

This project is licensed under the MIT License.
