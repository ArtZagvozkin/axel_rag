import os
from dotenv import load_dotenv
from logger import setup_logger

DISALLOWED_FILE_EXTENSIONS = [
    # Executables / binaries (Windows)
    ".exe", ".dll", ".sys", ".drv", ".com", ".scr", ".cpl", ".ocx",
    ".msi", ".msp", ".mst", ".pyd", ".mui", ".ax",

    # Executables (Linux / Unix)
    ".elf", ".so", ".ko", ".run",

    # macOS apps
    ".dmg", ".pkg", ".app",

    # Installers / packages
    ".apk", ".deb", ".rpm",

    # Disk / firmware / virtualization images
    ".iso", ".img", ".bin", ".rom",
    ".vhd", ".vhdx", ".qcow", ".qcow2", ".vmdk", ".vdi",
    ".efi",

    # Archives / compressed containers
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
    ".lz", ".lzma", ".lz4", ".zst", ".cab", ".ar", ".cpio",
    ".pak", ".pack",
]

logger = setup_logger()

load_dotenv()


# --- Env / tokens ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
XAI_MODEL = os.environ.get("XAI_MODEL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

if not XAI_API_KEY:
    raise RuntimeError("XAI_API_KEY is not set in .env")

if not XAI_MODEL:
    raise RuntimeError("XAI_MODEL is not set in .env")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

if not GROQ_MODEL:
    raise RuntimeError("GROQ_MODEL is not set in .env")

# --- Limits ---
MAX_HISTORY = 10
MAX_TELEGRAM_MESSAGE_LEN = 4000
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MiB
MEDIA_GROUP_TIMEOUT = 1.0

# --- LLM ---
GEMINI_MODEL = "gemini-2.5-flash-lite-preview-09-2025"

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

# --- Prompts ---
SYSTEM_PROMPT = """
Твоя роль: интеллектуальный помощник Arty AI.
Твоя цель: помогать пользователю в решении различных задач, предоставлять информацию и отвечать на вопросы.
"""

EMPTY_TRIGGER_FALLBACK_PROMPT = "Проанализируй последние сообщения и дай полезный ответ."

DEFAULT_IMAGE_PROMPT = "Твой ответ должен быть максимально естественным и валидным содержимому изображения. Нужно учитывать все возможные сведения из изображения."
DEFAULT_AUDIO_PROMPT = "Твой ответ должен быть максимально естественным и валидным содержимому аудио. Нужно учитывать все возможные сведения из аудио. Если в аудио есть вопрос, то на него нужно ответить. Если вопроса нет, то нужно максимально естественно поддержать разговор. Если нет речи, то нужно подробно и в деталях описать, что происходит в аудио."
DEFAULT_VIDEO_PROMPT = "Твой ответ должен быть максимально естественным и валидным содержимому видео. Нужно учитывать все возможные сведения из видео. Если в видео есть вопрос, то на него нужно ответить. Если вопроса нет, то нужно максимально естественно поддержать разговор. Если нет речи, то нужно подробно и в деталях описать, что происходит в видео."
DEFAULT_FILE_PROMPT = "Твой ответ должен быть максимально естественным и валидным содержимому файла. Нужно учитывать все возможные сведения из файла. Если в файле есть вопрос, то на него нужно ответить. Если вопроса нет, то нужно подробно и в деталях описать содержимое."

# --- RAG ---
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/rag_corpus")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "axel_chunks")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")

RAG_ENABLED = os.environ.get("RAG_ENABLED", "1") == "1"
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "6"))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "10000"))
