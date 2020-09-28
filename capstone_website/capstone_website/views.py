from flask import render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from .forms import LoginForm, RegisterForm
from .models import *

from capstone_website import app


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit() and request.method == 'POST':
        # user = User.query.filter_by(user=form.user.data.lower()).first()
        # if user is not None and user.verify_password(form.password.data):
        #     login_user(user, form.remember_me.data)
        #     return redirect(url_for('index'))
        # flash('Invalid email or password.')
        flash("Logged in!")
        return redirect(url_for('index'))
    return render_template("login.html", title="Login", form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit() and request.method == 'POST':
        # Create a new instance of User
        user = User(user=form.user.data, email=form.email.data, password=form.password.data)

        # Save user data into the database
        try:
            db.session.add(user)
            db.session.commit()
            # token = user.generate_confirmation_token()
            return redirect(url_for('login'))
        except:
            app.logger.info(f"Failed to save registration to database")

    return render_template("registration.html", title="Register", form=form)
