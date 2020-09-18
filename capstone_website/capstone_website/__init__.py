from flask import Flask

# Initialize the app
app = Flask(__name__)

# Load the views
from capstone_website import views

# Load the config file
app.config.from_object('config')