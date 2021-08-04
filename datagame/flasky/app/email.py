from threading import Thread
from flask import current_app, render_template
from flask_mail import Message
from . import mail
from flask import Flask
from flask_mail import Mail
from flask_mail import Message
import os

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
# app.config['smtp.UseDefaultCredentials'] = True
# app.config['smtp.EnableSsl'] = True
app.config['MAIL_USERNAME'] = '2739551399@qq.com'
app.config['MAIL_PASSWORD'] = 'giifgfmhxofdddhe'
app.config['FLASKY_MAIL_SUBJECT_PREFIX'] = '[Flasky]'
app.config['FLASKY_MAIL_SENDER'] = 'Flasky Admin <2739551399@qq.com>'
app.config['FLASKY_ADMIN'] = '2739551399@qq.com'
mail = Mail(app)

def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)


def send_email(to, subject, template, **kwargs):
    app = current_app._get_current_object()
    msg = Message(app.config['FLASKY_MAIL_SUBJECT_PREFIX'] + ' ' + subject,
                  sender=app.config['FLASKY_MAIL_SENDER'], recipients=[to])
    msg.body = render_template(template + '.txt', **kwargs)
    msg.html = render_template(template + '.html', **kwargs)
    thr = Thread(target=send_async_email, args=[app, msg])
    thr.start()
    return thr
