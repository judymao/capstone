from flask import Flask
from .config import *

# Initialize the app
app = Flask(__name__)

# Load the views
from capstone_website import views

# Load the config file
app.config.from_object(DevelopmentConfig)
