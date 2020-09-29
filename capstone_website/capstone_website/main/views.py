from flask import render_template, current_app, request, redirect, url_for, flash
from flask_login import login_required, current_user
from . import main
from ..models import User


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')