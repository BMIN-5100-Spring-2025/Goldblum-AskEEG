from flask import Flask, send_from_directory
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
    base_dir = os.getcwd()
    webapp_dir = os.path.join(base_dir, "Goldblum-AskEEG-Webapp")

    # Check if webapp directory exists
    if not os.path.exists(webapp_dir):
        logger.warning(f"Vue app directory not found at: {webapp_dir}")

    # Set static folder path to the Vue app's public directory
    public_dir = os.path.join(webapp_dir, "public")
    static_dir = os.path.join(public_dir, "static")
    static_output_dir = os.path.join(static_dir, "output")

    # Get path to data/output directory
    data_output_dir = os.path.join(base_dir, "data", "output")

    # Initialize Flask with explicit static folder pointing to Vue app's public directory
    app = Flask(__name__, static_folder=public_dir, static_url_path="/static")

    # Log static folder configuration
    logger.info(
        f"Flask app configured with static_folder={public_dir}, static_url_path=/static"
    )

    # Enable CORS with specific configuration
    CORS(
        app,
        resources={
            r"/api/*": {"origins": "*"},
            r"/static/*": {"origins": "*"},  # Explicitly allow CORS for static files
        },
    )
    logger.info("CORS configured to allow access to API and static resources")

    # Create data directories
    data_input_dir = os.path.join(base_dir, "data", "input")
    data_output_dir = os.path.join(base_dir, "data", "output")
    os.makedirs(data_input_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)
    logger.info(f"Ensured data input directory exists: {data_input_dir}")
    logger.info(f"Ensured data output directory exists: {data_output_dir}")

    # Add a direct route to serve files from data/output
    @app.route("/static/output/<path:filepath>")
    def serve_output_files(filepath):
        logger.info(f"Serving file from data/output: {filepath}")
        return send_from_directory(data_output_dir, filepath)

    # Check if we need to create the static directory structure
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        logger.info(f"Created static directory: {static_dir}")

    # Check if the symlink already exists and is correct
    is_correct_symlink = False
    if os.path.islink(static_output_dir):
        # Check if the symlink points to the correct location
        target = os.readlink(static_output_dir)
        if os.path.abspath(target) == os.path.abspath(data_output_dir):
            logger.info(
                f"Existing symlink is correctly configured: {static_output_dir} -> {data_output_dir}"
            )
            is_correct_symlink = True
        else:
            logger.warning(
                f"Existing symlink points to wrong location: {static_output_dir} -> {target}"
            )
            logger.warning(f"Expected: {data_output_dir}")
            # Remove the incorrect symlink
            os.unlink(static_output_dir)

    # If the symlink doesn't exist or was incorrect, create it
    if not is_correct_symlink and not os.path.exists(static_output_dir):
        try:
            os.symlink(data_output_dir, static_output_dir)
            logger.info(f"Created symlink: {static_output_dir} -> {data_output_dir}")
        except Exception as e:
            logger.error(f"Failed to create symlink: {e}")
            logger.warning("Static files may not be accessible!")

    # Register blueprints
    from app.backend.routes import api_bp, init_services

    app.register_blueprint(api_bp)

    # Register services initialization
    with app.app_context():
        init_services()

    return app
