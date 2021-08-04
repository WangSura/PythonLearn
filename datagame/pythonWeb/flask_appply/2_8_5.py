from logging import debug
from re import DEBUG
from flask import make_response
from flask import Flask, render_template
import flask
from werkzeug.utils import redirect
from werkzeug.wrappers import response
from flask import redirect
from flask import abort
app = Flask(__name__)


@app.route('/wrong')
def redir():
    return redirect('https://github.com')


'''

@app.route('/user/<id>')
def get_user(id):
    user = load_user(id)
    if not user:
        abort(404)
        return '<h1>Hello, {}</h1>'.format(user.name)


'''


@app.route('/')
def index():
    response = make_response('<h1>This documentaries carried a cookie</h1>')
    response.set_cookie('answer', '42')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
