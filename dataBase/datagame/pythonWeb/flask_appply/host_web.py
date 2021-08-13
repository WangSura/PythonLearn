from flask import Flask
from gevent import pywsgi

app = Flask(__name__)


@app.route('/report', methods=['get'])
def index():
    page = open(file_ikang, encoding='utf-8')
    res = page.read()
    return res


@app.route('/report_tjb', methods=['get'])
def index_1():
    page = open(file_tjb, encoding='utf-8')
    res = page.read()
    return res


server = pywsgi.WSGIServer(('0.0.0.0', 12345), app)
server.serve_forever()
