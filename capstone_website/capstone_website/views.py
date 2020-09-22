from flask import render_template, request, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from .forms import LoginForm

from capstone_website import app


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/register')
def register():
    return render_template("registration.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(form.email.data, form.remember_me.data))
        return redirect('/index')
    return render_template("login.html", form=form)


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        account = request.form['account']
        rating = request.form['rating']
        if account == '':
            return render_template('index.html', message='Please enter required fields.')
        return render_template("submit.html")
