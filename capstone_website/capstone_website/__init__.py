from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from config import *
import tiingo
import logging
import os

import chart_studio
chart_studio.tools.set_credentials_file(username=os.environ.get('PLOTLY_USER'),
                                        api_key=os.environ.get('PLOTLY_API'))


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

# Initialize
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)

# Set up email
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": os.environ['EMAIL_USER'],
    "MAIL_PASSWORD": os.environ['EMAIL_PASSWORD']
}

app.config.update(mail_settings)
mail = Mail(app)
mail.init_app(app)


# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

# Load Tiingo client
tiingo_config = {}
tiingo_config['session'] = True
tiingo_config['api_key'] = os.environ['TIINGO_API']  # StockConstants.API
client = tiingo.TiingoClient(tiingo_config)

quandl_api = os.environ["QUANDL_API"]
alpha_vantage_api = os.environ["ALPHA_VAN_API"]

from .main import main as main_blueprint
app.register_blueprint(main_blueprint)


from .auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint, url_prefix='/auth')

