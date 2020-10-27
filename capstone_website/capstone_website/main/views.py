from flask import render_template, current_app, request, redirect, url_for, flash
from flask_login import login_required, current_user
from . import main
from .forms import ContactForm
from ..models import User


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/contact_us')
def contact_us():
    form = ContactForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Successfully sent us an email!')
        return redirect(url_for('main.index'))

    return render_template('contact_us.html', title="Contact Us", form=form)


@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@main.route('/account')
@login_required
def account():
    return render_template('account.html')