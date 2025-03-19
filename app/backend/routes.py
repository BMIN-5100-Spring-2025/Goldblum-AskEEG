from flask import Blueprint, request, jsonify
from app.backend.process_query import process_query

main = Blueprint("main", __name__)


@main.route("/api/process", methods=["POST"])
def process():
    """
    Process a natural language query about EEG data
    """
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = process_query(query)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
