from flask import Flask
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("askeeg_backend")


def create_app():
    """Initialize and configure the Flask application"""
    app = Flask(__name__)

    # Enable CORS
    CORS(app)

    # Register blueprints
    from app.backend.routes import api_bp, init_services

    app.register_blueprint(api_bp)

    # Register services initialization
    with app.app_context():
        init_services()

    return app
