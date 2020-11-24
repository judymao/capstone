from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, validators, ValidationError, \
    IntegerField, RadioField
from wtforms.validators import DataRequired, Email, Length, Regexp
from wtforms.widgets import TextArea
from ..models import User, PortfolioInfo
from flask_login import current_user


class ContactForm(FlaskForm):
    name = StringField(label='Your Name', validators=[DataRequired(), Length(min=4, max=16),
                                                      Regexp('^[A-Za-z][A-Za-z0-9_.]*$', 0,
                                                             'Names must have only letters, numbers, dots or '
                                                             'underscores')])
    email = StringField(label='Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])
    subject = StringField(label='Subject', validators=[validators.DataRequired(), validators.Length(min=8, max=160)])
    message = StringField(label='Your Message', validators=[DataRequired(), Length(min=8)], widget=TextArea())


class ResetForm(FlaskForm):
    email = StringField(label='Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])


class Reset2Form(FlaskForm):
    email = StringField(label='Email', validators=[DataRequired(), Email(message='Invalid email'), Length(max=50)])
    token = StringField(label='8-Digit Token', validators=[validators.DataRequired(), validators.Length(
        min=8, max=8)])
    password = PasswordField('New Password', [validators.DataRequired(), validators.Length(min=8, max=80),
                                              validators.EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Password', validators=[DataRequired(), Length(min=8, max=80)])


class DeletePortfolio(FlaskForm):
    del_options = [(0, "Select an Option ..."), (1, "I am sure")]

    delete = SelectField(label='Are you sure you want to delete this portfolio? This cannot be undone.',
                         choices=del_options)

    def validate_delete(self, field):
        if field.data == '0':
            raise ValidationError('Please select an option from the dropdown menu.')


class RiskForm(FlaskForm):
    win_options = [(0, "A guaranteed gain of $50"), (1, '50% chance of gaining $100, 50% chance of gaining $0'),
                         (2, '1% chance of gaining $5000, 99% chance of gaining $0')]
    win = RadioField("Which of the following would you rather have?",
                                   choices=win_options, validators=[DataRequired()])

    lose_options = [(0, "A guaranteed loss of $50"), (1, '50% chance of losing $100, 50% chance of losing $0'),
                         (2, '1% chance of losing $5000, 99% chance of losing $0')]
    lose = RadioField("Which of the following would you rather have?",
                                       choices=lose_options, validators=[DataRequired()])

    chance_options = [(0, "Don't play"), (1, 'Play but gamble for low stakes'), (2, 'Sometimes go all-in')]
    chanceGames = RadioField("In games of chance, you:", choices=chance_options, validators=[DataRequired()])

    unknown_options = [(0, "Worries you a lot"), (1, '5Bothers you a bit, but you try to hope for the best'),
                         (2, 'Excites you')]
    unknownOutcomes = RadioField("The anticipation of events with an unknown outcome:",
                                   choices=unknown_options, validators=[DataRequired()])

    job_options = [(0, "A guaranteed loss of $50"), (1, '50% chance of losing $100, 50% chance of losing $0'),
                         (2, '1% chance of losing $5000, 99% chance of losing $0')]
    job = RadioField("Which of the following would you rather have?",
                                       choices=job_options, validators=[DataRequired()])

    monitor_options = [(0, "Investments doing poorly"), (1, 'All investments equally'), (2, 'Investments doing well')]
    monitorPortfolio = RadioField("When monitoring your portfolio, you focus on:", choices=chance_options, validators=[DataRequired()])


class PortfolioForm(FlaskForm):
    portfolioName = StringField('What would you like to name this portfolio?', validators=[DataRequired(),
                                                                                           Length(min=4, max=18),
                                                                                           Regexp(
                                                                                               '^[A-Za-z][A-Za-z0-9_.]*$',
                                                                                               0,
                                                                                               'Names must have only letters, numbers, dots or '
                                                                                               'underscores')])

    cash = IntegerField('How much are you willing to invest into this portfolio? Please enter a whole number.'
                        , validators=[DataRequired()])

    timeHorizon = IntegerField(label="How long do you intend to hold this portfolio for? Please enter your time "
                                     "horizon in number of years.", validators=[DataRequired()])

    def validate_portfolioName(self, field):
        user = User.query.filter_by(user=current_user.user).first()
        if PortfolioInfo.query.filter_by(user_id=user.id, name=field.data).first():
            raise ValidationError('Portfolio name already in use.')


class UpdateForm(FlaskForm):
    firstName = StringField('First Name', validators=[Length(min=2, max=80)])
    lastName = StringField('Last Name', validators=[Length(min=2, max=80)])
    email = StringField('Email', validators=[Email(message='Invalid email'), Length(max=50)])
    password = PasswordField('Password')
    confirm = PasswordField('Confirm Password')

    def validate_email(self, field):
        if current_user.email != field.data.lower():
            if User.query.filter_by(email=field.data.lower()).first():
                raise ValidationError('Email already registered.')

    def validate_password(self, field):
        if field.data:
            validators.Length(min=8, max=80)(self, field)
            validators.EqualTo('confirm', message='Passwords must match')(self, field)

    def validate_confirm(self, field):
        if field.data:
            validators.Length(min=8, max=80)(self, field)
