from flask import Blueprint, request, jsonify
from app.backend.process_query import process_query
from app.backend.tools import get_available_tools
import traceback

main = Blueprint("main", __name__)


@main.route("/api/process", methods=["POST"])
def process():
    """
    Process a natural language query about EEG data
    """
    try:
        data = request.json
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        result = process_query(query)
        return jsonify({"result": result})
    except Exception as e:
        # Get the full traceback for debugging
        error_traceback = traceback.format_exc()
        print(f"Error in /api/process: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


@main.route("/api/tools", methods=["GET"])
def get_tools():
    """
    Return the list of available tools
    """
    try:
        tools = get_available_tools()

        # Convert each Tool object to a dictionary for JSON serialization
        tools_data = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            tools_data.append(tool_dict)

        return jsonify({"tools": tools_data})
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in /api/tools: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500
