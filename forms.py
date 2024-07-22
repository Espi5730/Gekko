from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class userPrompt(FlaskForm):
    companyName = StringField('companyName',
                             validators=[DataRequired()])

    submit = SubmitField('Submit')

    def getName(self):
        return self.companyName.data