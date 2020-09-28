from flask import render_template, request, flash, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from .forms import LoginForm, RegisterForm
from . import auth
from ..models import User
from capstone_website import db


@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit() and request.method == 'POST':
        user = User.query.filter_by(user=form.user.data.lower()).first()
        if user is not None and user.verify_password(form.password.data):
            login_user(user, form.remember_me.data)
            return redirect(url_for('index'))
        flash('Invalid email or password.')
    return render_template('auth/login.html', title="Login", form=form)


@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit() and request.method == 'POST':
        # Create a new instance of User
        user = User(user=form.user.data, email=form.email.data, password=form.password.data)

        # Save user data into the database
        db.session.add(user)
        db.session.commit()
        # token = user.generate_confirmation_token()
        return redirect(url_for('auth.login'))

    return render_template('auth/registration.html', title="Register", form=form)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('index'))