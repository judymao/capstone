from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, validators, ValidationError, \
    IntegerField, FormField
from wtforms.validators import DataRequired, Email, Length, Regexp
from wtforms.widgets import TextArea
from wtforms.widgets import html5
from ..models import User, PortfolioInfo, PortfolioData
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


class RiskForm(FlaskForm):
    # 1 is risk adverse, 5 is high risk tolerance
    agreement_options = [(0, "Select an Option ..."), (1, 'Strongly Agree'), (2, 'Agree'), (3, 'Unsure'), (4,
                                                                                                           'Disagree'),
                         (5,
                          'Strongly '
                          'Disagree')]
    protectPortfolio = SelectField("Protecting my portfolio is more important to me than high returns.",
                                   choices=agreement_options)

    philosophy_options = [(0, "Select an Option ..."), (1, 'I feel comfortable with stable investments'), (2,
                                                                                                           'I am willing to withstand some '
                                                                                                           'fluctuations in my investment'),
                          (4, 'I am seeking substantial investment returns'),
                          (5, 'I am seeking potentially high investment returns')]
    investmentPhilosophy = SelectField("Which of the following statements best describes your investment philosophy?",
                                       choices=philosophy_options)

    expenditure_options = [(0, "Select an Option ..."), ('house', 'Buying a house'), ('tuition', 'Paying college '
                                                                                                 'tuition'),
                           ('venture', 'Capitalizing a new '
                                       'business venture'), ('retirement', 'Providing for my retirement')]
    expenditure = SelectField("What do you expect to be your next major expenditure?", choices=expenditure_options)

    def validate_protectPortfolio(self, field):
        if field.data == '0':
            raise ValidationError('Please select an option from the dropdown menu.')

    def validate_investmentPhilosophy(self, field):
        if field.data == '0':
            raise ValidationError('Please select an option from the dropdown menu.')

    def validate_expenditure(self, field):
        if field.data == '0':
            raise ValidationError('Please select an option from the dropdown menu.')


class PortfolioForm(FlaskForm):
    portfolioName = StringField('What would you like to name this portfolio?', validators=[DataRequired(),
                                                                                           Length(min=4, max=18),
                                                                                           Regexp(
                                                                                               '^[A-Za-z][A-Za-z0-9_.]*$',
                                                                                               0,
                                                                                               'Names must have only letters, numbers, dots or '
                                                                                               'underscores')])

    cash = IntegerField('How much are you willing to invest into this portfolio? Please enhow to ter a whole number.'
                        , validators=[DataRequired()])

    timeHorizon = IntegerField(label="How long do you intend to hold this portfolio for? Please enter your time "
                                     "horizon in number of years.", validators=[DataRequired()])

    def validate_portfolioName(self, field):
        user = User.query.filter_by(user=current_user.user).first()
        if PortfolioInfo.query.filter_by(user_id=user.id, name=field.data).first():
            raise ValidationError('Portfolio name already in use.')


class ConstraintForm(FlaskForm):
    long_options = [(-1, "Unsure"), (1, 'Yes'), (0, 'No')]

    longOnly = SelectField('Would you be willing to take on a short position, i.e. sell stocks first and then buy '
                           'back later?', choices=long_options)

    maxLeverage = IntegerField('How much leverage are you willing to take on, i.e. how much are you willing to borrow? '
                               'Please enter a whole number. If you have no maximum leverage constraint or are not '
                               'sure, enter 0.')

    maxAsset = IntegerField(label="Limit certain assets? How much? TBD QUESTION. If you are unsure, enter 0.")

    turnoverLimit = IntegerField(label="What is your turnover limit? If you are unsure, enter 0.")
