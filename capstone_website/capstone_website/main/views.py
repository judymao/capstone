from flask import render_template, current_app, request, redirect, url_for, flash
from flask_login import login_required, current_user
from . import main
from .forms import ContactForm, RiskForm, PortfolioForm, ResetForm, Reset2Form
from ..models import User


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/contact_us', methods=["GET", "POST"])
def contact_us():
    form = ContactForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Successfully sent us an email!')
        return redirect(url_for('main.index'))

    return render_template('contact_us.html', title="Contact Us", form=form)


@main.route('/reset', methods=["GET", "POST"])
def reset():
    form = ResetForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Please check your email')
        return redirect(url_for('auth.login'))

    return render_template('reset.html', title="Reset", form=form)


@main.route('/reset-password', methods=["GET", "POST"])
def reset2():
    form = Reset2Form()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Successfully reset password!')
        return redirect(url_for('auth.login'))

    return render_template('reset2.html', title="Reset Password", form=form)


@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@main.route('/portfolio/new-risk', methods=["GET", "POST"])
@login_required
def new_risk():
    form = RiskForm()
    if form.validate_on_submit() and request.method == 'POST':
        return redirect(url_for('main.new_general'))

    return render_template('new_portfolio_risk.html', title="New Portfolio - Risk", form=form)


@main.route('/portfolio/new-general', methods=["GET", "POST"])
@login_required
def new_general():
    form = PortfolioForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Successfully created a new Portfolio!')
        return redirect(url_for('main.dashboard'))

    return render_template('new_portfolio_general.html', title="New Portfolio - General", form=form)


@main.route('/account')
@login_required
def account():
    return render_template('account.html')