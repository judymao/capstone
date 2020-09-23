from flask import Flask
from .config import *
import logging
import os

# Initialize the app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load the views
from capstone_website import views

# Load config
mode = os.environ.get('CONFIG_STAGE', "PROD")
try:
    if mode == 'PROD':
        app.config.from_object(ProductionConfig)
        app.logger.info(f"Connected to prod")
    elif mode == 'DEV':
        app.config.from_object(DevelopmentConfig)
        app.logger.info(f"Connected to dev")
    else:
        logging.error(f"Cannot recognize config stage. Must be one of: [PROD, DEV]")

except ImportError:
    logging.error(f"Cannot import Config settings.")


