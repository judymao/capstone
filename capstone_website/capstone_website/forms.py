from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, validators, ValidationError
from wtforms.validators import DataRequired, Email, Length, Regexp
from .models import User


class LoginForm(FlaskForm):
    user = StringField('Username', validators=[DataRequired(), Length(min=4, max=16)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=80)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Log In')


class RegisterForm(FlaskForm):
    user = StringField('Username', validators=[DataRequired(), Length(min=4, max=16),
                                               Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
                                                      'Usernames must have only letters, numbers, dots or '
                                                      'underscores')])
    email = StringField('Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])
    password = PasswordField('Password', [validators.DataRequired(), validators.Length(min=8, max=80),
                                          validators.EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Password', validators=[DataRequired(), Length(min=8, max=80)])

    def validate_user(self, field):
        if User.query.filter_by(user=field.data).first():
            raise ValidationError('Username already in use.')

    def validate_email(self, field):
        if User.query.filter_by(email=field.data.lower()).first():
            raise ValidationError('Email already registered.')

