from flask import render_template, request, flash, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.urls import url_parse
from .forms import LoginForm, RegisterForm
from . import auth
from ..models import User
from capstone_website import db


@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    next_page = request.args.get('next')
    if form.validate_on_submit() and request.method == 'POST':
        user = User.query.filter_by(user=form.user.data).first()
        if user is None or not user.verify_password(form.password.data):
            flash('Invalid username or password.')
            return redirect(url_for('auth.login'))
        login_user(user)
        if next_page is None or not next_page.startswith('/'):
            next_page = url_for('main.dashboard')
        return redirect(next_page)
    return render_template('auth/login.html', title="Login", form=form)


@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegisterForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Create a new instance of User
        user = User(user=form.user.data, email=form.email.data, password=form.password.data)

        # Save user data into the database
        db.session.add(user)
        db.session.commit()

        flash('Successfully registered!')
        return redirect(url_for('auth.login'))

    return render_template('auth/registration.html', title="Register", form=form)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('main.index'))