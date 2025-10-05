# config.py
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///attendance.db")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")
SLACK_CHANNEL_NAME = os.getenv("SLACK_CHANNEL_NAME", "#attendance-alerts")
ORG_NAME = os.getenv("ORG_NAME", "Attendance System")
