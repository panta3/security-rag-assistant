from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./chroma_data"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"

    class Config:
        env_file = ".env"


settings = Settings()
