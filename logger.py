import logging
import os
import json
import requests
from datetime import datetime
import sys

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("AgenteRegulacion")

def log_event(event_type: str, data: dict):
    """
    Logs an event to stdout and optionally to a webhook.
    
    Args:
        event_type (str): The type of event (e.g., "query", "error", "feedback").
        data (dict): The data associated with the event.
    """
    timestamp = datetime.now().isoformat()
    
    # Structure the log entry
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "data": data
    }
    
    # Log to stdout (captured by DigitalOcean)
    logger.info(json.dumps(log_entry))
    
    # Send to webhook if configured
    webhook_url = os.environ.get("LOGGING_WEBHOOK_URL")
    if webhook_url:
        try:
            # Format depends on the destination (e.g., Slack, Discord, or generic JSON endpoint)
            # Sending raw JSON for maximum compatibility with generic ingestors
            response = requests.post(
                webhook_url, 
                json=log_entry,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            if response.status_code >= 400:
                logger.error(f"Failed to send log to webhook: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error sending log to webhook: {e}")
