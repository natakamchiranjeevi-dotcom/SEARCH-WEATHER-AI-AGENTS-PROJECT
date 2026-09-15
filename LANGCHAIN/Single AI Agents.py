import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #020617 100%
        );
        color: white;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #22d3ee
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .card-text {
        color: #cbd5e1;
        font-size: 14px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1e293b;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #334155;
        background: #1e293b;
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        color: #60a5fa;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 10px;
    }

    /* Status */
    .status {
        padding: 8px 12px;
        border-radius: 10px;
        background: #052e16;
        color: #86efac;
        font-size: 13px;
        text-align: center;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# CHECK API KEYS
# =========================================================

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing in your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.warning("⚠️ TAVILY_API_KEY is missing. Web search will not work.")

if not WEATHERSTACK_API_KEY:
    st.warning("⚠️ WEATHERSTACK_API_KEY is missing. Weather tool will not work.")


# =========================================================
# TOOLS
# =========================================================

search_tool = TavilySearch(
    max_results=3
)


@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not configured."

    try:

        url = (
            "http://api.weatherstack.com/current"
            f"?access_key={api_key}"
            f"&query={city}"
        )

        response = requests.get(
            url,
            timeout=10
        )

        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        humidity = data["current"]["humidity"]

        return (
            f"City: {city}\n"
            f"Temperature: {temperature}°C\n"
            f"Weather: {description}\n"
            f"Humidity: {humidity}%"
        )

    except Exception as e:

        return f"Weather service error: {str(e)}"


# =========================================================
# MODEL
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# =========================================================
# AGENT
# =========================================================

tools = [
    search_tool,
    get_weather
]

agent = create_agent(
    model=llm,
    tools=tools
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🤖 AI Agent")

    st.markdown(
        """
        <div class="status">
        ● Agent Online
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🛠️ Available Tools")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🌐 Web Search</div>
            <div class="card-text">
            Search the latest information from the internet.
            </div>
        </div>

        <div class="card">
            <div class="card-title">🌤️ Weather</div>
            <div class="card-text">
            Get current weather information for any city.
            </div>
        </div>

        <div class="card">
            <div class="card-title">🧠 Groq LLM</div>
            <div class="card-text">
            Powered by GPT-OSS-20B through Groq.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ Controls")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        <div class="footer">
        Single AI Agent System<br>
        Built with LangChain + Groq + Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN UI
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI Agent Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions, search the web, or check live weather using AI.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# WELCOME CARDS
# =========================================================

if not st.session_state.messages:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🌐 Web Search</div>
                <div class="card-text">
                Ask about the latest news, technology,
                events and information.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🌤️ Weather</div>
                <div class="card-text">
                Ask questions such as:
                "What's the weather in Ongole?"
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🧠 AI Chat</div>
                <div class="card-text">
                Have a normal conversation with
                your intelligent AI agent.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "💬 Ask your AI agent anything..."
)


if prompt:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })

                answer = response["messages"][-1].content

            except Exception as e:

                answer = (
                    "❌ **Something went wrong.**\n\n"
                    f"`{str(e)}`"
                )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })