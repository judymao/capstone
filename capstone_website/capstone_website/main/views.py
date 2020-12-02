from flask import render_template, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from flask_mail import Message
from . import main
from .forms import ContactForm, RiskForm, PortfolioForm, ResetForm, Reset2Form, DeletePortfolio, UpdateForm
from ..models import User, PortfolioInfo, PortfolioData, Stock
from capstone_website import db, mail, app
from timeit import default_timer as timer
from capstone_website.src.constants import Constants

import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objects as go
import pandas as pd

from chart_studio.exceptions import PlotlyRequestError


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
    portfolio_info = PortfolioInfo()
    portfolios = portfolio_info.get_portfolios(user_id=user.id)
    portfolio_table = create_portfolio_summary(portfolios)

    return render_template('dashboard.html', portfolios=portfolios, table=portfolio_table)


@main.route('/portfolio/<portfolio_name>')
@login_required
def portfolio_page(portfolio_name):

    constants = Constants()

    user = User.query.filter_by(user=current_user.user).first()
    portfolio_info = PortfolioInfo()
    portfolio_data = PortfolioData()
    portfolios = portfolio_info.get_portfolios(user_id=user.id)
    curr_portfolio = portfolio_info.get_portfolio_instance(user_id=user.id, portfolio_name=portfolio_name)

    portfolio_data_df = portfolio_data.get_portfolio_data_df(user_id=user.id, portfolio_id=curr_portfolio.id)
    spy_df = Stock.get_etf(constants.SPY, portfolio_data_df.iloc[0]["date"], portfolio_data_df.iloc[-1]["date"])
    portfolio_graph = create_portfolio_graph(portfolio_data_df, spy_df, portfolio_name)
    portfolio_pie = create_portfolio_pie(portfolio_data_df)
    portfolio_table = create_portfolio_table(portfolio_data_df, curr_portfolio)

    # TO-DO: replace this with portfolio table
    # portfolio_table = create_portfolio_summary(portfolios)

    return render_template('portfolio.html', portfolios=portfolios, curr_portfolio=curr_portfolio,
                           portfolio_graph=portfolio_graph, pie_graph=portfolio_pie, table=portfolio_table)


@main.route('/portfolio/<portfolio_name>/delete', methods=["GET", "POST"])
@login_required
def delete_portfolio(portfolio_name):
    form = DeletePortfolio()
    user = User.query.filter_by(user=current_user.user).first()
    curr_portfolio = PortfolioInfo.query.filter_by(user_id=user.id, name=portfolio_name).first()

    if form.validate_on_submit() and request.method == 'POST':
        PortfolioData.query.filter_by(user_id=user.id, portfolio_id=curr_portfolio.id).delete()
        PortfolioInfo.query.filter_by(user_id=user.id, name=portfolio_name).delete()
        db.session.commit()
        flash('Portfolio ' + portfolio_name + ' deleted!')

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

        start = timer()
        # portfolio_info = PortfolioInfo.query.filter_by(user_id=user.id, name=form.portfolioName.data).first()
        portfolio_info = portfolio.get_portfolio_instance(user_id=user.id, portfolio_name=form.portfolioName.data)
        portfolio_data = portfolio_info.create_portfolio()
        end = timer()
        print(f"Creating portfolio took: {end - start} seconds")

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
        return redirect(url_for('main.portfolio_page', portfolio_name=form.portfolioName.data))

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


# Helper Function Below
def create_portfolio_graph(portfolio, spy, portf_name):
    # print(portfolio)
    if portfolio.shape[0]:
        # Render a graph and return the URL
        # layout = go.Layout(yaxis=dict(tickformat=".2%"))

        spy["close"] = spy["close"] * portfolio.iloc[0]["value"] / spy.iloc[0]["close"]
        spy = spy.sort_values(by="date").groupby("date").last().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio["date"], y=portfolio["value"], mode="lines", name="Portfolio Value")) #, layout=layout)
        fig.add_trace(go.Scatter(x=spy["date"], y=spy["close"], mode="lines", name="SPY")) #, layout=layout)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Portfolio Value')
        portfolio_graph_url = get_portfolio_graph_url(fig)
        while portfolio_graph_url is None:
            portfolio_graph_url = get_portfolio_graph_url(fig)
        print(portfolio_graph_url)
        plot_html = tls.get_embed(portfolio_graph_url)

        return plot_html


def get_portfolio_graph_url(fig, name=1):
    portfolio_graph_url = None
    while portfolio_graph_url is None:
        try:
            portfolio_graph_url = py.plot(fig, filename=f"portfolio_value_{name}", auto_open=False, )
        except PlotlyRequestError:
            print(f"Ran into PlotlyRequestError. Trying new filename")
            name += 1
    return portfolio_graph_url


def create_portfolio_pie(portfolio):

    if portfolio.shape[0]:
        df = pd.DataFrame({"assets": portfolio.iloc[-1]["assets"],
                           "weights": portfolio.iloc[-1]["weights"]
                           })
        df = df[df["weights"] > 0]
        fig = go.Figure(data=go.Pie(labels=df["assets"], values=df["weights"]))
        fig_url = py.plot(fig, filename="portfolio_pie", auto_open=False, )
        plot_html = tls.get_embed(fig_url)
        print(fig_url)
        return plot_html

def create_portfolio_table(portfolio, portfolio_info):

    if portfolio.shape[0]:
        df = pd.DataFrame({"Returns": [f"{portfolio_info.returns:,.2%}" if portfolio_info.returns is not None else "NA"],
                           "Volatility": [f"{portfolio_info.volatility:,.2%}" if portfolio_info.volatility is not None else "NA"],
                           "Sharpe Ratio": [f"{portfolio_info.sharpe_ratio:,.2f}" if portfolio_info.sharpe_ratio is not None else "NA"]
                           }).transpose().reset_index().rename(columns={"index": "Metric", 0: "Value"})
        table_html = df.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                           '<table class="table">')
        table_html = table_html.replace("text-align: right;", "text-align: left;")
        table_html = table_html.replace('<thead>', '<thead class="thead-dark">')
        print(table_html)
        return table_html

def create_portfolio_summary(portfolios):
    portfolios_list = portfolios.all()

    names, time_horizons, investments, returns, curr_values = [], [], [], [], []
    for portfolio in portfolios_list:
        names += [portfolio.name]
        time_horizons += [int(portfolio.time_horizon)]
        investments += ['$' + str(portfolio.cash)]
        returns += [f"{portfolio.returns:,.2%}" if portfolio.returns is not None else "NA"]
        curr_values += [f"${((1 + portfolio.returns) * portfolio.cash):,.2f}" if portfolio.returns is not None else "NA"]

    summary_df = pd.DataFrame({"Portfolio Name": names, "Time Horizon (Years)": time_horizons,
                               "Initial Investment Amount": investments,
                               "Current Portfolio Value": curr_values,
                               "Return": returns
                               })
    summary_html = summary_df.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                           '<table class="table">')
    summary_html = summary_html.replace("text-align: right;", "text-align: left;")
    summary_html = summary_html.replace('<thead>', '<thead class="thead-dark">')

    print(summary_html)
    return summary_html