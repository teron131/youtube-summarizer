import logging
import sys


def setup_logging():
    """Configure structured logging for the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    # Ensure stdout is flushed for Railway/Docker
    sys.stdout.reconfigure(line_buffering=True)
