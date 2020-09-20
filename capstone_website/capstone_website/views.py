from flask import render_template, request
from flask_sqlalchemy import SQLAlchemy

from capstone_website import app


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        account = request.form['account']
        rating = request.form['rating']
        if account == '':
            return render_template('index.html', message='Please enter required fields.')
        return render_template("submit.html")
