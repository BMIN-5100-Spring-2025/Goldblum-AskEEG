import os
import re
import json
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from .tools import (
    get_available_tools,
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("process_query")

# Load environment variables
load_dotenv()

# Azure OpenAI environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Regular OpenAI API key as fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def llm_analyze_query(query: str) -> dict:
    """
    Use LLM to analyze the query and determine parameters

    Args:
        query: The user's natural language query

    Returns:
        dict: Extracted parameters and analysis type
    """
    try:
        # Get available tools as text
        tools = get_available_tools()
        tools_text = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are an expert EEG analysis assistant that helps analyze EEG data.
            
            The user will ask questions about their EEG data. Your job is to extract parameters from their request:
            1. Which tool would be most appropriate for their request
            2. What time window they're interested in
            3. What frequency band (if any) they want to analyze
            4. Any specific channels they mentioned
            
            Available tools:
            {tools_text}
            
            Respond with a JSON object containing:
            1. tool_name: The tool to use
            2. time_window: [start_time, end_time] in seconds
            3. frequency_band: Band to analyze (delta, theta, alpha, beta, gamma) or null
            4. channel_ids: Array of channel IDs or null if not specified
            
            Human: {query}
            
            Assistant: """
        )

        # Try to use Azure OpenAI first
        response = None
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            try:
                from langchain_openai import AzureChatOpenAI

                llm = AzureChatOpenAI(
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version=AZURE_OPENAI_API_VERSION,
                    model=AZURE_OPENAI_MODEL,
                    temperature=0,
                )

                response = llm.invoke(prompt.format(query=query, tools_text=tools_text))
            except Exception as azure_error:
                logger.error(f"Azure OpenAI error: {str(azure_error)}")

        # Use OpenAI as fallback
        if not response and OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model="gpt-4o-mini",
                    temperature=0,
                )

                response = llm.invoke(prompt.format(query=query, tools_text=tools_text))
            except Exception as openai_error:
                logger.error(f"OpenAI error: {str(openai_error)}")
                return {"error": f"Error with OpenAI API: {str(openai_error)}"}

        if not response:
            return {"error": "Failed to process query. API keys not configured."}

        # Extract JSON from the response
        try:
            content = response.content

            # Check if response is wrapped in markdown code blocks
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, content)
            if match:
                # Extract the JSON content from the code block
                content = match.group(1)

            # Parse the JSON
            result = json.loads(content)
            logger.info(f"LLM returned valid JSON: {result}")
            return result
        except json.JSONDecodeError:
            # Log the error instead of falling back silently
            logger.error(f"Failed to parse JSON from LLM response: {response.content}")
            return {"error": "Failed to parse LLM response as JSON"}

    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}", exc_info=True)
        return {"error": str(e)}
