"""
Application Settings with YAML + Environment Variable Support

Priority order:
1. Environment variables (highest)
2. .env file  
3. config/default.yaml
4. Hardcoded defaults (lowest)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_prompts_config() -> Dict[str, str]:
    """Load prompts from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "prompts.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """
    Application Settings using Pydantic.
    
    Loads from environment variables and .env file.
    YAML config is used for defaults that can be overridden.
    """
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore',
        env_nested_delimiter='__'
    )

    # -------------------------------------------------------------------------
    # Milvus / Zilliz Cloud
    # -------------------------------------------------------------------------
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: str = Field(default="19530", alias="MILVUS_PORT")
    milvus_user: str = Field(default="", alias="MILVUS_USER")
    milvus_password: str = Field(default="", alias="MILVUS_PASSWORD")
    milvus_secure: bool = Field(default=False, alias="MILVUS_SECURE")
    milvus_collection_name: str = Field(default="legal_documents", alias="MILVUS_COLLECTION_NAME")
    
    # HNSW Index Parameters
    milvus_hnsw_m: int = Field(default=8, alias="MILVUS_HNSW_M")
    milvus_hnsw_ef_construction: int = Field(default=100, alias="MILVUS_HNSW_EF_CONSTRUCTION")
    milvus_hnsw_ef_search: int = Field(default=32, alias="MILVUS_HNSW_EF_SEARCH")

    # -------------------------------------------------------------------------
    # BGE-M3 Embedding
    # -------------------------------------------------------------------------
    bge_model_name: str = Field(default="BAAI/bge-m3", alias="BGE_MODEL_NAME")
    bge_max_length: int = Field(default=1024, alias="BGE_MAX_LENGTH")
    bge_use_fp16: bool = Field(default=False, alias="BGE_USE_FP16")
    bge_batch_size: int = Field(default=4, alias="BGE_BATCH_SIZE")

    # -------------------------------------------------------------------------
    # Reranker
    # -------------------------------------------------------------------------
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_NAME")
    reranker_top_k: int = Field(default=100, alias="RERANKER_TOP_K")
    reranker_top_n: int = Field(default=5, alias="RERANKER_TOP_N")
    reranker_batch_size: int = Field(default=64, alias="RERANKER_BATCH_SIZE")
    enable_reranker: bool = Field(default=False, alias="ENABLE_RERANKER")

    # -------------------------------------------------------------------------
    # Gemini (Primary LLM)
    # -------------------------------------------------------------------------
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # -------------------------------------------------------------------------
    # Groq (Fallback LLM)
    # -------------------------------------------------------------------------
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # -------------------------------------------------------------------------
    # DeepSeek (Alternative LLM)
    # -------------------------------------------------------------------------
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # -------------------------------------------------------------------------
    # LlamaCloud (OCR/Parsing)
    # -------------------------------------------------------------------------
    llama_cloud_api_key: Optional[str] = Field(default=None, alias="LLAMA_CLOUD_API_KEY")

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # -------------------------------------------------------------------------
    # Generation Defaults
    # -------------------------------------------------------------------------
    generation_temperature: float = Field(default=0.1, alias="GENERATION_TEMPERATURE")
    generation_max_tokens: int = Field(default=1024, alias="GENERATION_MAX_TOKENS")
    
    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------
    max_chunk_chars: int = Field(default=2000, alias="MAX_CHUNK_CHARS")
    min_chunk_chars: int = Field(default=100, alias="MIN_CHUNK_CHARS")

    @property
    def is_zilliz_cloud(self) -> bool:
        """Check if using Zilliz Cloud (vs local Milvus)."""
        return self.milvus_secure and "cloud.zilliz.com" in self.milvus_host

    def get_prompts(self) -> Dict[str, str]:
        """Load prompts from YAML config."""
        return load_prompts_config()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
