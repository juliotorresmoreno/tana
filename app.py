from flask import Flask
from endpoints.ai import router_ai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.register_blueprint(router_ai, url_prefix='/ai')

