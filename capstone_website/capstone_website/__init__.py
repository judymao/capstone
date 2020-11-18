from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import *
import logging
import os

# Create the app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

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

# Initialize database
db = SQLAlchemy(app)
Bootstrap(app)


# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'


from .main import main as main_blueprint
app.register_blueprint(main_blueprint)


from .auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint, url_prefix='/auth')

