from flask import render_template, current_app, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from . import main
from .forms import ContactForm, RiskForm, PortfolioForm, ResetForm, Reset2Form, DeletePortfolio
from ..models import User, PortfolioInfo, PortfolioData
from capstone_website import db


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


@main.route('/portfolio/<portfolio_name>/delete', methods=["GET", "POST"])
@login_required
def delete_portfolio(portfolio_name):
    form = DeletePortfolio()
    user = User.query.filter_by(user=current_user.user).first()
    curr_portfolio = PortfolioInfo.query.filter_by(user_id=user.id, name=portfolio_name).first()

    if form.validate_on_submit() and request.method == 'POST':
        PortfolioInfo.query.filter_by(user_id=user.id, name=portfolio_name).delete()
        db.session.commit()
        flash('Portfolio '+ portfolio_name+ ' deleted!')

        portfolios = PortfolioInfo.query.filter_by(user_id=user.id)
        return redirect(url_for('main.dashboard', portfolios=portfolios))
    return render_template('delete_portfolio.html', curr_portfolio=curr_portfolio, form=form)


@main.route('/portfolio/new-risk', methods=["GET", "POST"])
@login_required
def new_risk():
    form = RiskForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Store form results in session variables

        session['win'] = form.win.data
        session['lose'] = form.lose.data
        session['game'] = form.chanceGames.data
        session['job'] = form.job.data
        session['unknown'] = form.unknownOutcomes.data
        session['monitor'] = form.monitorPortfolio.data

        return redirect(url_for('main.new_general')) # Go to the next set of questions

    return render_template('new_portfolio_risk.html', title="New Portfolio - Risk", form=form)


@main.route('/portfolio/new-general', methods=["GET", "POST"])
@login_required
def new_general():
    form = PortfolioForm()
    if form.validate_on_submit() and request.method == 'POST':
        # Get the user details
        user = User.query.filter_by(user=current_user.user).first()

        # Create a new instance of Portfolio
        portfolio = PortfolioInfo(user_id=user.id, win_philosophy=session['win'],
                                  lose_philosophy=session['lose'], games_philosophy=session['game'],
                                  unknown_philosophy=session['unknown'], job_philosophy=session['job'],
                                  monitor_philosophy=session['monitor'], name=form.portfolioName.data,
                                  time_horizon=form.timeHorizon.data, cash=form.cash.data)

        # Save portfolio data into the database
        db.session.add(portfolio)
        db.session.commit()

        # Remove the session variables
        session.pop('loss', None)
        session.pop('win', None)
        session.pop('game', None)
        session.pop('unknown', None)
        session.pop('job', None)
        session.pop('monitor', None)

        flash('Successfully created a new Portfolio!')
        return redirect(url_for('main.dashboard'))

    return render_template('new_portfolio_general.html', title="New Portfolio - General", form=form)


@main.route('/account')
@login_required
def account():
    return render_template('account/account.html')