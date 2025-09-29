from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, Field
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    POSTGRES_HOST: str = Field("192.168.0.221", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(5432, env="POSTGRES_PORT")
    POSTGRES_USER: str = Field("postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("postgres", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("coj_db", env="POSTGRES_DB")

    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __post_init_post_parse__(self):
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()
