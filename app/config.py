import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python -m app.server")