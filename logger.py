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

def _get_valid_webhook_url() -> str | None:
    """Returns the webhook URL only if it has a valid scheme, otherwise None."""
    url = os.environ.get("LOGGING_WEBHOOK_URL", "").strip()
    if url and (url.startswith("http://") or url.startswith("https://")):
        return url
    return None

def log_event(event_type: str, data: dict):
    """
    Logs an event to stdout and optionally to a webhook.

    Args:
        event_type (str): The type of event (e.g., "query", "error", "feedback").
        data (dict): The data associated with the event.
    """
    timestamp = datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "data": data
    }

    # Log to stdout (captured by DigitalOcean)
    logger.info(json.dumps(log_entry))

    # Send to webhook only if a valid URL is configured
    webhook_url = _get_valid_webhook_url()
    if webhook_url:
        try:
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
