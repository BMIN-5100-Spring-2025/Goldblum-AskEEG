import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from app.backend.tools import get_available_tools

# Load environment variables
load_dotenv()

# Azure OpenAI environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Regular OpenAI API key as fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def process_query(query: str) -> str:
    """
    Process a natural language query about EEG data

    Args:
        query: The user's natural language query

    Returns:
        A string response to the user
    """
    try:
        # Get available tools as text
        tools = get_available_tools()
        tools_text = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        # Create a prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are an expert EEG analysis assistant called AskEEG that helps analyze EEG data.
            
            The user will ask questions about their EEG data. Your job is to determine:
            1. Which tool would be most appropriate for their request
            2. What time window they're interested in
            3. Any other parameters they've specified
            
            Available tools:
            {tools_text}
            
            Respond in a conversational way with:
            1. The tool you would use
            2. The time window mentioned
            3. Any other parameters (like frequency bands, channels, sensitivity)
            
            Human: {query}
            
            Assistant: """
        )

        # Try to use Azure OpenAI first
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
                return response.content
            except Exception as azure_error:
                print(f"Azure OpenAI error: {str(azure_error)}")
                # Fall through to try regular OpenAI

        # Use regular OpenAI as fallback
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model="gpt-3.5-turbo",
                    temperature=0,
                )

                response = llm.invoke(prompt.format(query=query, tools_text=tools_text))
                return response.content
            except Exception as openai_error:
                return f"Error with OpenAI API: {str(openai_error)}"

        # If neither Azure nor regular OpenAI is available, return a mock response
        tool_name = "Unknown"
        time_window = "Unknown"

        if "synchrony" in query.lower() or "brain regions" in query.lower():
            tool_name = "synchrony_analyzer"
        elif (
            "alpha" in query.lower()
            or "delta" in query.lower()
            or "sleep" in query.lower()
        ):
            tool_name = "alpha_delta_ratio"
        elif (
            "spike" in query.lower()
            or "seizure" in query.lower()
            or "epilep" in query.lower()
        ):
            tool_name = "spike_detector"

        return f"I would use the {tool_name} tool to analyze your EEG data. (Note: This is a mock response as no API keys are configured.)"

    except Exception as e:
        return f"Error processing your query: {str(e)}"
