from flask import Flask, render_template
from gevent import pywsgi
from flask_bootstrap import Bootstrap
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)


bootstrap = Bootstrap(app)
if __name__ == '__main__':
    #server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    # server.serve_forever()
    app.run(host='0.0.0.0', debug=True)
