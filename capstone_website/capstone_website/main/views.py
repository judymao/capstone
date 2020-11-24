from flask import render_template, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from flask_mail import Message
from . import main
from .forms import ContactForm, RiskForm, PortfolioForm, ResetForm, Reset2Form, DeletePortfolio, UpdateForm
from ..models import User, PortfolioInfo, PortfolioData
from capstone_website import db, mail, app

@main.route('/')
def index():
    return render_template("index.html")


@main.route('/contact_us', methods=["GET", "POST"])
def contact_us():
    form = ContactForm()
    if form.validate_on_submit() and request.method == 'POST':

        msg = Message(subject=form.subject.data, recipients=[app.config.get("MAIL_USERNAME")],
                      sender=app.config.get("MAIL_USERNAME"),
                      reply_to=form.email.data, body=form.message.data)

        # Send the mail
        mail.send(msg)

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
        session.pop('loss', None)
        session.pop('win', None)
        session.pop('game', None)
        session.pop('unknown', None)
        session.pop('job', None)
        session.pop('monitor', None)

        flash('Successfully created a new Portfolio!')
        return redirect(url_for('main.dashboard'))

    return render_template('new_portfolio_general.html', title="New Portfolio - General", form=form)


@main.route('/account', methods=["GET", "POST"])
@login_required
def account():
    # Get the user details
    user = User.query.filter_by(user=current_user.user).first()

    form = UpdateForm()

    if request.method == 'GET':
        form.firstName.data = user.first_name
        form.lastName.data = user.last_name
        form.email.data = user.email

    if request.method == 'POST' and form.validate_on_submit():
        user.first_name = form.firstName.data
        user.last_name = form.lastName.data

        if form.email.data:
            user.email = form.email.data

        if form.password.data:
            user.password = form.password.data

        db.session.commit()

        # Clear sensitive information
        form.password.data = ""
        form.confirm.data = ""

        flash("Successfully updated your account information!")
        return render_template('account/account.html', form=form)

    return render_template('account/account.html', form=form)
