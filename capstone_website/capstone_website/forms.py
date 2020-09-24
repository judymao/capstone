from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, validators
from wtforms.validators import DataRequired, Email, Length


class LoginForm(FlaskForm):
    user = StringField('Username', validators=[DataRequired(), Length(min=4, max=16)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=80)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegisterForm(FlaskForm):
    user = StringField('Username', validators=[DataRequired(), Length(min=4, max=16)])
    email = StringField('Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])
    password = PasswordField('Password', [validators.DataRequired(), validators.Length(min=8, max=80),
                                          validators.EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Password', validators=[DataRequired(), Length(min=8, max=80)])
