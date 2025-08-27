import os
from dotenv import load_dotenv

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_config():
    load_dotenv()
    sp_scopes = os.getenv("SP_SCOPES")
    scopes = sp_scopes.split(",") if sp_scopes else ["https://graph.microsoft.com/.default"]
    return Config(
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
        SP_CLIENT_ID=os.getenv("SHAREPOINT_CLIENT_ID"),
        SP_CLIENT_SECRET=os.getenv("SHAREPOINT_CLIENT_SECRET"),
        SP_AUTHORITY=os.getenv("SHAREPOINT_AUTHORITY"),
        SP_SITE_URL=os.getenv("SHAREPOINT_SITE_URL"),
        SP_SCOPES=scopes,
        EMBEDDING_MODEL=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        BASE_DIR=os.getenv("BASE_DIR", "resources/vectorstore"),
        SP_COLLECTION=os.getenv("SP_COLLECTION", "sharepoint_rfp"),
        UP_COLLECTION=os.getenv("UP_COLLECTION", "uploaded_rfp"),
        PROMPT_TRUNCATE=int(os.getenv("PROMPT_TRUNCATE", "10000")),
        ANALYSIS_MODEL=os.getenv("ANALYSIS_MODEL", "gpt-4.1"),
        UNDERSTAND_MODEL=os.getenv("UNDERSTAND_MODEL", "gpt-4.1"),
        CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "1000"))
    )