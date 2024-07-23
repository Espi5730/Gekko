from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class addPortfolio(FlaskForm):
    adding = HiddenField('please')

    submit = SubmitField('Add to Portfolio')