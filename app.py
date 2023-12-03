from flask import Flask
from flask_cors import CORS
from endpoints.generate import router_ai
from endpoints.embeddings import router_embeddings
from endpoints.conversation import router_conversation

app = Flask(__name__)
CORS(app)

app.register_blueprint(router_ai, url_prefix='/generate')
app.register_blueprint(router_embeddings, url_prefix='/embeddings')
app.register_blueprint(router_conversation, url_prefix='/conversation')
