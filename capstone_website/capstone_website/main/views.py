from flask import render_template, current_app, request, redirect, url_for, flash
from flask_login import login_required, current_user
from . import main
from ..models import User


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/profile/<username>')
@login_required
def user(username):
    user = User.query.filter_by(user=username).first_or_404()
    return render_template('user.html', user=user)