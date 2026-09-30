import os
import math
from pathlib import Path
from typing import TypedDict, Annotated

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Command Center",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FIND .ENV FILE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

possible_env_files = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]

env_loaded = False

for env_path in possible_env_files:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        env_loaded = True
        break

if not env_loaded:
    load_dotenv()


# ============================================================
# API KEYS
# ============================================================

MISTRAL_API_KEY = os.getenv(
    "MISTRAL_API_KEY",
    ""
).strip()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
).strip()

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY",
    ""
).strip()

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY",
    ""
).strip()

WEATHERSTACK_API_KEY = os.getenv(
    "WEATHERSTACK_API_KEY",
    ""
).strip()

ALPHA_API_KEY = os.getenv(
    "ALPHA_API_KEY",
    ""
).strip()

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY",
    ""
).strip()


# ============================================================
# GLOBAL CSS
#
# IMPORTANT:
# Custom HTML is rendered with st.html()
# NOT st.markdown()
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --background: #03060d;
        --panel: rgba(10, 17, 31, 0.90);
        --panel-light: rgba(15, 25, 44, 0.82);
        --border: rgba(91, 151, 255, 0.16);
        --border-bright: rgba(91, 151, 255, 0.32);

        --text: #edf5ff;
        --muted: #7f8da5;

        --blue: #6da9ff;
        --blue-light: #a7caff;

        --green: #3ee59d;
        --orange: #ffb86b;
        --purple: #8b5cf6;
    }


    html,
    body,
    [class*="css"] {
        font-family: 'Inter', sans-serif;
    }


    .stApp {

        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(45, 104, 255, 0.16),
                transparent 30%
            ),

            radial-gradient(
                circle at 92% 4%,
                rgba(123, 73, 255, 0.14),
                transparent 28%
            ),

            radial-gradient(
                circle at 50% 100%,
                rgba(25, 80, 170, 0.08),
                transparent 30%
            ),

            linear-gradient(
                135deg,
                #02040a 0%,
                #07101e 52%,
                #02050b 100%
            );

        color: var(--text);
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    #MainMenu,
    footer {
        visibility: hidden;
    }


    .block-container {

        max-width: 1500px;

        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                rgba(2, 7, 15, 0.99),
                rgba(3, 8, 16, 0.99)
            );

        border-right:
            1px solid rgba(100, 160, 255, 0.10);
    }


    section[data-testid="stSidebar"]
    .block-container {

        padding:
            1.4rem
            1rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .command-header {

        position: relative;

        display: flex;

        align-items: center;

        justify-content: space-between;

        padding:
            25px
            28px;

        margin-bottom: 20px;

        border:
            1px solid var(--border);

        border-radius: 23px;

        background:
            linear-gradient(
                135deg,
                rgba(14, 28, 51, 0.94),
                rgba(5, 11, 22, 0.86)
            );

        box-shadow:
            0 25px 70px rgba(0,0,0,0.28),

            inset
            0 0 40px
            rgba(80, 145, 255, 0.035);

        overflow: hidden;
    }


    .command-header::before {

        content: "";

        position: absolute;

        top: 0;

        left: 7%;

        width: 86%;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(104, 169, 255, 0.9),
                transparent
            );
    }


    .command-header::after {

        content: "";

        position: absolute;

        bottom: 0;

        left: 18%;

        width: 64%;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(120, 91, 255, 0.35),
                transparent
            );
    }


    .brand-area {

        display: flex;

        align-items: center;

        gap: 17px;
    }


    /* ========================================================
       AI CORE
       ======================================================== */

    .core {

        position: relative;

        width: 60px;
        height: 60px;

        min-width: 60px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 50%;

        color: white;

        font-size: 23px;

        background:
            radial-gradient(
                circle,
                #eef8ff 0%,
                #81b9ff 20%,
                #4b7ee8 43%,
                #1c3570 68%,
                #050a15 100%
            );

        box-shadow:

            0 0 12px
            rgba(92, 165, 255, 0.95),

            0 0 35px
            rgba(60, 130, 255, 0.48),

            inset
            0 0 16px
            rgba(255,255,255,0.35);
    }


    .core-ring {

        position: absolute;

        width: 76px;
        height: 76px;

        border-radius: 50%;

        border:
            1px solid
            rgba(100, 170, 255, 0.18);

        animation:
            pulse 2.8s infinite ease-in-out;
    }


    @keyframes pulse {

        0% {
            transform: scale(0.94);
            opacity: 0.35;
        }

        50% {
            transform: scale(1.06);
            opacity: 0.75;
        }

        100% {
            transform: scale(0.94);
            opacity: 0.35;
        }
    }


    .title {

        font-family:
            'Space Grotesk',
            sans-serif;

        font-size: 28px;

        font-weight: 700;

        letter-spacing: -0.7px;

        color:
            var(--text);
    }


    .subtitle {

        margin-top: 5px;

        color:
            var(--muted);

        font-size: 12px;
    }


    .online {

        display: flex;

        align-items: center;

        gap: 8px;

        color:
            #91f2c2;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1px;
    }


    .online-dot {

        width: 8px;
        height: 8px;

        border-radius: 50%;

        background:
            var(--green);

        box-shadow:

            0 0 8px
            rgba(62,229,157,1),

            0 0 20px
            rgba(62,229,157,0.45);
    }


    /* ========================================================
       SECTION LABEL
       ======================================================== */

    .section-label {

        margin-bottom: 9px;

        color:
            #72829c;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1.7px;

        text-transform: uppercase;
    }


    /* ========================================================
       GLASS CARD
       ======================================================== */

    .glass-card {

        border:
            1px solid var(--border);

        border-radius:
            18px;

        padding:
            18px;

        background:
            linear-gradient(
                145deg,
                rgba(16, 29, 51, 0.84),
                rgba(5, 12, 23, 0.78)
            );

        box-shadow:
            0 15px 45px
            rgba(0,0,0,0.18);

        backdrop-filter:
            blur(15px);
    }


    /* ========================================================
       TOOL CARD
       ======================================================== */

    .tool-card {

        padding:
            13px
            14px;

        margin-bottom:
            8px;

        border:
            1px solid
            rgba(110,168,255,0.11);

        border-radius:
            13px;

        background:
            rgba(255,255,255,0.025);
    }


    .tool-name {

        color:
            #dce9ff;

        font-size:
            13px;

        font-weight:
            600;
    }


    .tool-desc {

        margin-top:
            4px;

        color:
            var(--muted);

        font-size:
            10px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-pill {

        display:
            inline-flex;

        align-items:
            center;

        gap:
            7px;

        padding:
            7px
            12px;

        border:
            1px solid
            var(--border);

        border-radius:
            999px;

        background:
            rgba(255,255,255,0.035);

        color:
            #b7c4d9;

        font-size:
            10px;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {

        border:
            1px solid
            rgba(110,168,255,0.10);

        border-radius:
            17px;

        background:
            rgba(8,15,28,0.72);

        margin-bottom:
            10px;
    }


    [data-testid="stChatInput"] {

        border-color:
            rgba(110,168,255,0.28)
            !important;
    }


    div[data-testid="stChatInput"]
    textarea {

        background:
            rgba(5,10,19,0.96)
            !important;

        color:
            #edf5ff
            !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        border:
            1px solid
            rgba(110,168,255,0.17);

        border-radius:
            11px;

        background:
            rgba(255,255,255,0.035);

        color:
            #dce8ff;
    }


    .stButton > button:hover {

        border-color:
            rgba(110,168,255,0.55);

        background:
            rgba(80,130,255,0.10);
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {

        border:
            1px solid
            rgba(110,168,255,0.10);

        border-radius:
            15px;

        background:
            rgba(255,255,255,0.025);

        padding:
            13px;
    }


    /* ========================================================
       CHAT AVATAR CUSTOMIZATION
       ======================================================== */

    [data-testid="stChatMessageAvatarUser"] {

        background:
            rgba(50,100,190,0.18);

        border:
            1px solid
            rgba(100,170,255,0.20);
    }


    [data-testid="stChatMessageAvatarAssistant"] {

        background:
            radial-gradient(
                circle,
                rgba(120,180,255,0.35),
                rgba(30,60,120,0.30)
            );

        border:
            1px solid
            rgba(100,170,255,0.24);
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .command-header {

            padding:
                17px;
        }

        .title {

            font-size:
                20px;
        }

        .online {

            display:
                none;
        }

        .core {

            width:
                48px;

            height:
                48px;

            min-width:
                48px;
        }
    }

    </style>
    """
)


# ============================================================
# HELPER FOR HTML
# ============================================================

def render_html(content: str):
    """
    Render custom HTML safely using Streamlit's HTML renderer.

    IMPORTANT:
    Do not replace this with st.markdown().
    """
    st.html(content)


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            allowed,
        )

        return str(result)

    except Exception as e:

        return (
            f"Calculation error: {e}"
        )


# ============================================================
# STOCK TOOL
# ============================================================

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get the latest stock quote using Alpha Vantage.
    """

    if not ALPHA_API_KEY:

        return {
            "error":
                "ALPHA_API_KEY is not configured "
                "in the .env file."
        }

    try:

        response = requests.get(
            "https://www.alphavantage.co/query",

            params={
                "function":
                    "GLOBAL_QUOTE",

                "symbol":
                    symbol.upper().strip(),

                "apikey":
                    ALPHA_API_KEY,
            },

            timeout=15,
        )

        response.raise_for_status()

        data = response.json()

        return data

    except requests.RequestException as e:

        return {
            "error":
                f"Stock service error: {e}"
        }


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_current_weather(location: str) -> str:
    """
    Get current weather using OpenWeather.
    """

    if not OPENWEATHER_API_KEY:

        return (
            "OPENWEATHER_API_KEY is not configured "
            "in the .env file."
        )

    try:

        # ----------------------------------------------------
        # GEOCODING
        # ----------------------------------------------------

        geo_response = requests.get(

            "https://api.openweathermap.org/geo/1.0/direct",

            params={
                "q":
                    location,

                "limit":
                    1,

                "appid":
                    OPENWEATHER_API_KEY,
            },

            timeout=10,
        )

        geo_response.raise_for_status()

        locations = geo_response.json()

        if not locations:

            return (
                f"Could not find location: "
                f"{location}"
            )

        location_data = locations[0]

        latitude = location_data["lat"]

        longitude = location_data["lon"]

        city_name = location_data.get(
            "name",
            location,
        )

        country = location_data.get(
            "country",
            "",
        )

        state = location_data.get(
            "state",
            "",
        )


        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        weather_response = requests.get(

            "https://api.openweathermap.org/data/2.5/weather",

            params={

                "lat":
                    latitude,

                "lon":
                    longitude,

                "appid":
                    OPENWEATHER_API_KEY,

                "units":
                    "metric",
            },

            timeout=10,
        )

        weather_response.raise_for_status()

        data = weather_response.json()


        # ----------------------------------------------------
        # VALUES
        # ----------------------------------------------------

        condition = (
            data["weather"][0]["description"]
            .title()
        )

        temperature = (
            data["main"]["temp"]
        )

        feels_like = (
            data["main"]["feels_like"]
        )

        humidity = (
            data["main"]["humidity"]
        )

        pressure = (
            data["main"]["pressure"]
        )

        wind_speed = (
            data.get("wind", {})
            .get("speed", "N/A")
        )

        visibility = (
            data.get("visibility")
        )

        if visibility is not None:

            visibility_km = round(
                visibility / 1000,
                1,
            )

        else:

            visibility_km = "N/A"


        # ----------------------------------------------------
        # LOCATION NAME
        # ----------------------------------------------------

        location_parts = [
            city_name
        ]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        resolved_location = ", ".join(
            location_parts
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return (
            f"Current weather in "
            f"{resolved_location}:\n\n"

            f"Condition: "
            f"{condition}\n"

            f"Temperature: "
            f"{temperature}°C\n"

            f"Feels like: "
            f"{feels_like}°C\n"

            f"Humidity: "
            f"{humidity}%\n"

            f"Pressure: "
            f"{pressure} hPa\n"

            f"Wind speed: "
            f"{wind_speed} m/s\n"

            f"Visibility: "
            f"{visibility_km} km"
        )


    except requests.Timeout:

        return (
            "Weather service request timed out."
        )


    except requests.HTTPError as e:

        status = (
            e.response.status_code
            if e.response
            else "unknown"
        )

        if status == 401:

            return (
                "OpenWeather API key is "
                "invalid or inactive."
            )

        return (
            f"Weather API HTTP error: "
            f"{status}"
        )


    except requests.RequestException as e:

        return (
            "Could not connect to weather "
            f"service: {e}"
        )


    except Exception as e:

        return (
            f"Weather error: {e}"
        )


# ============================================================
# TAVILY SEARCH TOOL
# ============================================================

def build_search_tool():

    if not TAVILY_API_KEY:

        return None

    try:

        return TavilySearch(

            max_results=5,

            topic="general",

            search_depth="advanced",
        )

    except Exception:

        return None


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# BUILD LANGGRAPH AGENT
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def build_agent():

    if not GROQ_API_KEY:

        return None


    try:

        # ----------------------------------------------------
        # GROQ MODEL
        # ----------------------------------------------------

        llm = ChatGroq(

            model="openai/gpt-oss-20b",

            api_key=GROQ_API_KEY,

            temperature=0.2,
        )


        # ----------------------------------------------------
        # TOOLS
        # ----------------------------------------------------

        tools = [

            calculator,

            get_stock_price,

            get_current_weather,
        ]


        # ----------------------------------------------------
        # TAVILY
        # ----------------------------------------------------

        search_tool = (
            build_search_tool()
        )

        if search_tool:

            tools.insert(
                0,
                search_tool,
            )


        # ----------------------------------------------------
        # BIND TOOLS
        # ----------------------------------------------------

        llm_with_tools = (
            llm.bind_tools(tools)
        )


        # ----------------------------------------------------
        # CHAT NODE
        # ----------------------------------------------------

        def chat_node(
            state: ChatState
        ):

            response = (
                llm_with_tools.invoke(
                    state["messages"]
                )
            )

            return {
                "messages":
                    [response]
            }


        # ----------------------------------------------------
        # TOOL NODE
        # ----------------------------------------------------

        tool_node = ToolNode(
            tools
        )


        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        graph = StateGraph(
            ChatState
        )


        graph.add_node(
            "chat",
            chat_node,
        )


        graph.add_node(
            "tools",
            tool_node,
        )


        graph.add_edge(
            START,
            "chat",
        )


        graph.add_conditional_edges(
            "chat",
            tools_condition,
        )


        graph.add_edge(
            "tools",
            "chat",
        )


        # ----------------------------------------------------
        # COMPILE
        # ----------------------------------------------------

        return graph.compile()


    except Exception as e:

        st.session_state[
            "agent_build_error"
        ] = str(e)

        return None


# ============================================================
# CREATE AGENT
# ============================================================

agent = build_agent()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "request_count" not in st.session_state:

    st.session_state.request_count = 0


if "last_tool" not in st.session_state:

    st.session_state.last_tool = "None"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div style="
            font-family:'Space Grotesk';
            font-size:21px;
            font-weight:700;
            color:#edf5ff;
        ">
            ◈ COMMAND CENTER
        </div>

        <div style="
            color:#7e8ca5;
            font-size:10px;
            margin-top:4px;
        ">
            Intelligent Tool Orchestration
        </div>
        """
    )


    st.write("")


    render_html(
        """
        <div class="section-label">
            SYSTEM
        </div>
        """
    )


    if agent is not None:

        render_html(
            """
            <span class="status-pill">

                <span style="
                    color:#3ee59d;
                    text-shadow:
                    0 0 10px #3ee59d;
                ">
                    ●
                </span>

                SYSTEM ONLINE

            </span>
            """
        )

    else:

        render_html(
            """
            <span class="status-pill">

                <span style="
                    color:#ffb86b;
                ">
                    ●
                </span>

                CONFIGURATION REQUIRED

            </span>
            """
        )


    st.write("")


    render_html(
        """
        <div class="section-label">
            AVAILABLE TOOLS
        </div>
        """
    )


    sidebar_tools = [

        (
            "⌕",
            "Web Search",
            "Tavily real-time search",
        ),

        (
            "∑",
            "Calculator",
            "Mathematical reasoning",
        ),

        (
            "↗",
            "Stock Price",
            "Alpha Vantage",
        ),

        (
            "☁",
            "Weather",
            "OpenWeather",
        ),
    ]


    for icon, name, description in sidebar_tools:

        render_html(
            f"""
            <div class="tool-card">

                <div class="tool-name">

                    {icon}
                    &nbsp;&nbsp;
                    {name}

                </div>

                <div class="tool-desc">

                    {description}

                </div>

            </div>
            """
        )


    st.write("")


    render_html(
        """
        <div class="section-label">
            SESSION
        </div>
        """
    )


    st.metric(
        "Requests",
        st.session_state.request_count,
    )


    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.request_count = 0

        st.session_state.last_tool = "None"

        st.rerun()


    render_html(
        """
        <div style="
            margin-top:18px;
            color:#65728a;
            font-size:10px;
            line-height:1.6;
        ">

            LangGraph routes the request
            through the AI model and invokes
            tools when required.

            <br><br>

            API keys remain in your local
            environment.

        </div>
        """
    )


# ============================================================
# MAIN HEADER
# ============================================================

render_html(
    """
    <div class="command-header">

        <div class="brand-area">

            <div class="core">

                <div class="core-ring"></div>

                ◈

            </div>


            <div>

                <div class="title">

                    Intelligent AI Command Center

                </div>


                <div class="subtitle">

                    Natural language
                    • Tool reasoning
                    • Real-time information

                </div>

            </div>

        </div>


        <div class="online">

            <span class="online-dot"></span>

            SYSTEM ONLINE

        </div>

    </div>
    """
)


# ============================================================
# TOP STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Architecture",
        "LangGraph",
    )


with col2:

    st.metric(
        "LLM",
        "Groq",
    )


with col3:

    st.metric(
        "Tools",
        "4",
    )


with col4:

    st.metric(
        "Last Tool",
        st.session_state.last_tool,
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns(
    [2.7, 1],
    gap="large",
)


# ============================================================
# LEFT - CONVERSATION
# ============================================================

with left:

    render_html(
        """
        <div class="section-label">
            CONVERSATION
        </div>
        """
    )


    # --------------------------------------------------------
    # WELCOME PANEL
    # --------------------------------------------------------

    if not st.session_state.messages:

        render_html(
            """
            <div class="glass-card">

                <div style="
                    font-family:'Space Grotesk';
                    font-size:23px;
                    font-weight:700;
                    color:#edf5ff;
                ">

                    Ready for your command.

                </div>


                <div style="
                    color:#7e8ca5;
                    font-size:12px;
                    margin-top:8px;
                    line-height:1.7;
                ">

                    Ask naturally.
                    The AI will decide when
                    an external tool is required.

                </div>

            </div>
            """
        )


        st.write("")


        q1, q2, q3 = st.columns(3)


        with q1:

            render_html(
                """
                <div class="glass-card">

                    <div style="
                        font-weight:600;
                        font-size:13px;
                    ">

                        ⌕ Search

                    </div>


                    <div style="
                        color:#7e8ca5;
                        font-size:10px;
                        margin-top:5px;
                    ">

                        Current information

                    </div>

                </div>
                """
            )


        with q2:

            render_html(
                """
                <div class="glass-card">

                    <div style="
                        font-weight:600;
                        font-size:13px;
                    ">

                        ∑ Calculate

                    </div>


                    <div style="
                        color:#7e8ca5;
                        font-size:10px;
                        margin-top:5px;
                    ">

                        Mathematical problems

                    </div>

                </div>
                """
            )


        with q3:

            render_html(
                """
                <div class="glass-card">

                    <div style="
                        font-weight:600;
                        font-size:13px;
                    ">

                        ☁ Weather

                    </div>


                    <div style="
                        color:#7e8ca5;
                        font-size:10px;
                        margin-top:5px;
                    ">

                        Live weather

                    </div>

                </div>
                """
            )


        st.write("")


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    #
    # IMPORTANT:
    # NO avatar="●"
    # NO avatar="◈"
    # --------------------------------------------------------

    for message in st.session_state.messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            role = "user"

        elif isinstance(
            message,
            AIMessage,
        ):

            role = "assistant"

        else:

            continue


        content = message.content


        if not isinstance(
            content,
            str,
        ):

            content = str(content)


        with st.chat_message(role):

            st.markdown(content)


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    prompt = st.chat_input(
        "Enter your command...",
    )


    # --------------------------------------------------------
    # PROCESS REQUEST
    # --------------------------------------------------------

    if prompt:

        if agent is None:

            st.error(
                "AI agent is not available."
            )


            if not GROQ_API_KEY:

                st.warning(
                    "GROQ_API_KEY is missing "
                    "from your .env file."
                )


            elif "agent_build_error" in st.session_state:

                st.code(
                    st.session_state[
                        "agent_build_error"
                    ]
                )


        else:

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            user_message = HumanMessage(
                content=prompt
            )


            st.session_state.messages.append(
                user_message
            )


            st.session_state.request_count += 1


            # ------------------------------------------------
            # DISPLAY USER
            #
            # NO CUSTOM AVATAR ARGUMENT
            # ------------------------------------------------

            with st.chat_message(
                "user"
            ):

                st.markdown(prompt)


            # ------------------------------------------------
            # AI RESPONSE
            # ------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "AI is processing..."
                ):

                    try:

                        result = agent.invoke(
                            {
                                "messages":
                                    st.session_state.messages
                            }
                        )


                        new_messages = (
                            result["messages"]
                        )


                        # ------------------------------------
                        # FIND LAST TOOL
                        # ------------------------------------

                        detected_tool = None


                        for msg in new_messages:

                            if (
                                getattr(
                                    msg,
                                    "type",
                                    "",
                                )
                                == "tool"
                            ):

                                detected_tool = (
                                    getattr(
                                        msg,
                                        "name",
                                        None,
                                    )
                                )

                                if detected_tool:

                                    break


                        if detected_tool:

                            st.session_state.last_tool = (
                                detected_tool
                            )


                        # ------------------------------------
                        # FINAL RESPONSE
                        # ------------------------------------

                        final_message = (
                            new_messages[-1]
                        )


                        if isinstance(
                            final_message,
                            AIMessage,
                        ):

                            answer = (
                                final_message.content
                            )

                        else:

                            answer = str(
                                final_message.content
                            )


                        if not isinstance(
                            answer,
                            str,
                        ):

                            answer = str(answer)


                        st.markdown(
                            answer
                        )


                        # ------------------------------------
                        # SAVE FULL GRAPH HISTORY
                        # ------------------------------------

                        st.session_state.messages = (
                            new_messages
                        )


                    except Exception as e:

                        st.error(
                            "Agent execution error"
                        )

                        st.code(
                            str(e)
                        )


# ============================================================
# RIGHT - AGENT MONITOR
# ============================================================

with right:

    render_html(
        """
        <div class="section-label">
            AGENT MONITOR
        </div>
        """
    )


    current_state = (
        "READY"
        if agent
        else "CONFIG REQUIRED"
    )


    render_html(
        f"""
        <div class="glass-card">

            <div style="
                color:#71819b;
                font-size:9px;
                letter-spacing:1.5px;
            ">

                CURRENT STATE

            </div>


            <div style="
                font-family:'Space Grotesk';
                font-size:20px;
                font-weight:700;
                margin-top:7px;
            ">

                {current_state}

            </div>


            <div style="
                height:1px;
                background:
                    rgba(110,168,255,.10);
                margin:16px 0;
            ">
            </div>


            <div style="
                color:#71819b;
                font-size:9px;
                letter-spacing:1.5px;
            ">

                LAST TOOL

            </div>


            <div style="
                font-size:13px;
                font-weight:600;
                margin-top:6px;
            ">

                {st.session_state.last_tool}

            </div>

        </div>
        """
    )


    st.write("")


    # ========================================================
    # EXECUTION FLOW
    # ========================================================

    render_html(
        """
        <div class="section-label">
            EXECUTION FLOW
        </div>
        """
    )


    render_html(
        """
        <div class="glass-card">

            <div class="tool-card">

                <div class="tool-name">
                    START
                </div>

                <div class="tool-desc">
                    User command
                </div>

            </div>


            <div style="
                text-align:center;
                color:#5d6c84;
            ">
                ↓
            </div>


            <div class="tool-card">

                <div class="tool-name">
                    AI NODE
                </div>

                <div class="tool-desc">
                    Understand request
                </div>

            </div>


            <div style="
                text-align:center;
                color:#5d6c84;
            ">
                ↓
            </div>


            <div class="tool-card">

                <div class="tool-name">
                    TOOL ROUTER
                </div>

                <div class="tool-desc">
                    Select required tool
                </div>

            </div>


            <div style="
                text-align:center;
                color:#5d6c84;
            ">
                ↓
            </div>


            <div class="tool-card">

                <div class="tool-name">
                    TOOL EXECUTION
                </div>

                <div class="tool-desc">
                    Search • Math • Stock • Weather
                </div>

            </div>


            <div style="
                text-align:center;
                color:#5d6c84;
            ">
                ↺
            </div>


            <div class="tool-card">

                <div class="tool-name">
                    AI RESPONSE
                </div>

                <div class="tool-desc">
                    Generate final answer
                </div>

            </div>

        </div>
        """
    )


    st.write("")


    # ========================================================
    # API STATUS
    # ========================================================

    render_html(
        """
        <div class="section-label">
            API STATUS
        </div>
        """
    )


    api_status = [

        (
            "Groq",
            GROQ_API_KEY,
        ),

        (
            "Tavily",
            TAVILY_API_KEY,
        ),

        (
            "OpenWeather",
            OPENWEATHER_API_KEY,
        ),

        (
            "Alpha Vantage",
            ALPHA_API_KEY,
        ),

        (
            "Mistral",
            MISTRAL_API_KEY,
        ),

        (
            "Google",
            GOOGLE_API_KEY,
        ),
    ]


    for name, key in api_status:

        if key:

            symbol = "●"

            status = "READY"

            status_color = "#3ee59d"

        else:

            symbol = "○"

            status = "NOT CONFIGURED"

            status_color = "#65728a"


        render_html(
            f"""
            <div style="
                display:flex;
                justify-content:space-between;
                padding:7px 2px;
                font-size:11px;
            ">

                <span>

                    <span style="
                        color:{status_color};
                    ">

                        {symbol}

                    </span>

                    &nbsp;

                    {name}

                </span>


                <span style="
                    color:{status_color};
                    font-size:9px;
                ">

                    {status}

                </span>

            </div>
            """
        )


    st.write("")


    # ========================================================
    # ENVIRONMENT INFO
    # ========================================================

    env_display = "Detected" if env_loaded else "Not detected"


    render_html(
        f"""
        <div class="glass-card">

            <div style="
                color:#71819b;
                font-size:9px;
                letter-spacing:1.4px;
            ">

                ENVIRONMENT

            </div>


            <div style="
                color:#aebbd0;
                font-size:10px;
                margin-top:8px;
            ">

                .env status:
                {env_display}

            </div>

        </div>
        """
    )