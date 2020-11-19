from flask import render_template, current_app, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from . import main
from .forms import ContactForm, RiskForm, PortfolioForm, ConstraintForm, ResetForm, Reset2Form
from ..models import User, PortfolioInfo, PortfolioData
from capstone_website import db
from datetime import date


@main.route('/')
def index():
    return render_template("index.html")


@main.route('/contact_us', methods=["GET", "POST"])
def contact_us():
    form = ContactForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Successfully sent us an email!')
        return redirect(url_for('main.index'))

    return render_template('account/contact_us.html', title="Contact Us", form=form)


@main.route('/reset', methods=["GET", "POST"])
def reset():
    form = ResetForm()
    if form.validate_on_submit() and request.method == 'POST':

        flash('Please check your email')
        return redirect(url_for('auth.login'))

    return render_template('account/reset.html', title="Reset", form=form)


@main.route('/reset-password', methods=["GET", "POST"])
def reset2():
    form = Reset2Form()
    if form.validate_on_submit() and request.method == 'POST':

        # Get the user details
        user = User.query.filter_by(user=form.user.data).first()
        user.password = form.password.data # Update the user's password in the database
        db.session.commit()

        flash('Successfully reset password!')
        return redirect(url_for('auth.login'))

    return render_template('account/reset2.html', title="Reset Password", form=form)


@main.route('/dashboard')
@login_required
def dashboard():
    user = User.query.filter_by(user=current_user.user).first()
    portfolios = PortfolioInfo.query.filter_by(user_id=user.id)

    return render_template('dashboard.html', portfolios=portfolios)


@main.route('/portfolio/<portfolio_name>')
@login_required
def portfolio(portfolio_name):

    user = User.query.filter_by(user=current_user.user).first()
    portfolios = PortfolioInfo.query.filter_by(user_id=user.id)
    curr_portfolio = PortfolioInfo.query.filter_by(user_id=user.id, name=portfolio_name).first()

    return render_template('portfolio.html', portfolios=portfolios, curr_portfolio=curr_portfolio)


@main.route('/portfolio/new-risk', methods=["GET", "POST"])
@login_required
def new_risk():
    form = RiskForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Store form results in session variables

        session['protect_portfolio'] = form.protectPortfolio.data
        session['inv_philosophy'] = form.investmentPhilosophy.data
        session['next_expenditure'] = form.expenditure.data

        return redirect(url_for('main.new_general')) # Go to the next set of questions

    return render_template('new_portfolio_risk.html', title="New Portfolio - Risk", form=form)


@main.route('/portfolio/new-general', methods=["GET", "POST"])
@login_required
def new_general():
    form = PortfolioForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Store form results in session variables

        session['portfolio_name'] = form.portfolioName.data
        session['time_horizon'] = form.timeHorizon.data

        return redirect(url_for('main.new_specific'))

    return render_template('new_portfolio_general.html', title="New Portfolio - General", form=form)


@main.route('/portfolio/new-specific', methods=["GET", "POST"])
@login_required
def new_specific():
    form = ConstraintForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Get the user details
        user = User.query.filter_by(user=current_user.user).first()

        # Create a new instance of Portfolio

        # Stores the metadata for the portfolio (static variables)
        portfolio = PortfolioInfo(user_id=user.id, protect_portfolio=session['protect_portfolio'],
                              inv_philosophy=session['inv_philosophy'], next_expenditure=session['next_expenditure'],
                              name=session['portfolio_name'], time_horizon=session['time_horizon'])

        # Save portfolio info into the database
        db.session.add(portfolio)
        db.session.commit()

        # Generate a portfolio given the portfolio info
        #TODO: Rather than pulling from PostgreSQL again, is there a way to get the portfolio_id before storing portfolio_info?
        portfolio_info = PortfolioInfo.query.filter_by(user_id=user.id, name=session['portfolio_name']).first()
        portfolio_graph, portfolio_data = portfolio_info.create_portfolio()

        # Save portfolio data into the database
        db.session.add_all(portfolio_data)
        db.session.commit()

        # Remove the session variables
        session.pop('protect_portfolio', None)
        session.pop('inv_philosophy', None)
        session.pop('next_expenditure', None)
        session.pop('portfolio_name', None)
        session.pop('time_horizon', None)

        flash('Successfully created a new Portfolio!')
        return redirect(url_for('main.dashboard'))

    return render_template('new_portfolio_specific.html', title="New Portfolio - Constraints", form=form)


@main.route('/account')
@login_required
def account():
    return render_template('account/account.html')