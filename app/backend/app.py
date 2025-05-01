import os
from dotenv import load_dotenv
from app.backend import create_app

# Load environment variables
load_dotenv()

# Create Flask application
app = create_app()

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))

    # Start Flask development server
    app.run(host="0.0.0.0", port=port, debug=True)
