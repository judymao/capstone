from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, validators, ValidationError
from wtforms.validators import DataRequired, Email, Length, Regexp
from wtforms.widgets import TextArea


class ContactForm(FlaskForm):
    name = StringField('Your Name', validators=[DataRequired(), Length(min=4, max=16),
                                               Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
                                                      'Names must have only letters, numbers, dots or '
                                                      'underscores')])
    email = StringField('Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])
    subject = StringField('Subject', [validators.DataRequired(), validators.Length(min=8, max=160)])
    message = StringField('Your Message', validators=[DataRequired(), Length(min=8)], widget=TextArea())
