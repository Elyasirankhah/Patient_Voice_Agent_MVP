"""
ECHO AI Agent - Streamlit App
AI-powered analysis of patient-provider communication patterns
Uses ML/LLM for accurate analysis with personalized coaching
"""
import streamlit as st
import json
import re
from pathlib import Path
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import pandas as pd
import requests

# Try to import new modules
try:
    from evaluator_eppc import EPPCEvaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False

try:
    from thread_analyzer import ThreadAnalyzer
    HAS_THREAD_ANALYZER = True
except ImportError:
    HAS_THREAD_ANALYZER = False

try:
    from chat_agent import ChatAgent
    HAS_CHAT_AGENT = True
except ImportError:
    HAS_CHAT_AGENT = False

try:
    from communication_coach import CommunicationCoach
    HAS_COACH = True
except ImportError:
    HAS_COACH = False

try:
    from response_generator import ResponseGenerator
    HAS_RESPONSE_GEN = True
except ImportError:
    HAS_RESPONSE_GEN = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# OpenAI API Key - try multiple sources (for deployment flexibility)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
# Try Streamlit secrets as fallback (for cloud deployment)
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', '')
    except:
        pass

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in .env file or Streamlit secrets")
else:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Try to import ML analyzer
try:
    from analyzer_ml import MLAnalyzer
    HAS_ML = True
except ImportError:
    HAS_ML = False
    st.warning("⚠️ ML analyzer not available. Install: pip install sentence-transformers")

# Try to import LLM analyzer
try:
    from analyzer_llm import LLMAnalyzer
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Try to import trained model analyzer
try:
    from analyzer_trained import TrainedAnalyzer
    HAS_TRAINED = True
except ImportError:
    HAS_TRAINED = False

# Try to import voice transcription
try:
    from voice.transcription import transcribe_audio_bytes
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

# Page config
st.set_page_config(
    page_title="ECHO AI Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize analyzers
@st.cache_resource
def get_ml_analyzer():
    """Initialize ML analyzer (cached)"""
    if HAS_ML:
        try:
            # Load codebooks first
            eppc, sdoh = load_codebooks()
            if not eppc or not sdoh:
                return None
            
            # Save to temporary files for analyzer
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            eppc_path = os.path.join(temp_dir, 'codebook_eppc.json')
            sdoh_path = os.path.join(temp_dir, 'codebook_sdoh.json')
            
            with open(eppc_path, 'w', encoding='utf-8') as f:
                json.dump(eppc, f)
            with open(sdoh_path, 'w', encoding='utf-8') as f:
                json.dump(sdoh, f)
            
            analyzer = MLAnalyzer(eppc_path, sdoh_path)
            return analyzer
        except Exception as e:
            import traceback
            print(f"Error initializing ML analyzer: {e}")
            traceback.print_exc()
            return None
    return None

@st.cache_resource
def get_llm_analyzer():
    """Initialize LLM analyzer (cached) - API key hidden"""
    if HAS_LLM:
        try:
            # Load codebooks first
            eppc, sdoh = load_codebooks()
            if not eppc or not sdoh:
                return None
            
            # Save to temporary files for analyzer
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            eppc_path = os.path.join(temp_dir, 'codebook_eppc.json')
            sdoh_path = os.path.join(temp_dir, 'codebook_sdoh.json')
            
            with open(eppc_path, 'w', encoding='utf-8') as f:
                json.dump(eppc, f)
            with open(sdoh_path, 'w', encoding='utf-8') as f:
                json.dump(sdoh, f)
            
            return LLMAnalyzer(eppc_path, sdoh_path, OPENAI_API_KEY)
        except Exception as e:
            st.error(f"Error initializing LLM analyzer: {e}")
            return None
    return None

@st.cache_resource
def get_trained_analyzer():
    """Initialize trained model analyzer (cached)"""
    if HAS_TRAINED:
        try:
            # Get HF token from secrets
            hf_token = st.secrets.get('HF_TOKEN', os.getenv('HF_TOKEN', ''))
            # Load from Hugging Face
            return TrainedAnalyzer('Elyasirankhah/patient-voice-biobert', hf_token=hf_token)
        except Exception as e:
            st.error(f"Error initializing trained analyzer: {e}")
            return None
    return None

@st.cache_resource
def get_eppc_evaluator():
    """Initialize EPPC evaluator (cached)"""
    if HAS_EVALUATOR:
        try:
            return EPPCEvaluator()
        except Exception as e:
            return None
    return None

@st.cache_resource
def get_thread_analyzer():
    """Initialize thread analyzer (cached)"""
    if HAS_THREAD_ANALYZER:
        try:
            def analyze_func(text):
                return analyze_text(text, analyzer_type="llm", threshold=0.5)
            evaluator = get_eppc_evaluator()
            return ThreadAnalyzer(analyze_func, evaluator)
        except Exception as e:
            return None
    return None

@st.cache_resource
def get_chat_agent():
    """Initialize chat agent (cached)"""
    if HAS_CHAT_AGENT:
        try:
            return ChatAgent(OPENAI_API_KEY)
        except Exception as e:
            return None
    return None

@st.cache_resource
def get_communication_coach():
    """Initialize communication coach (cached)"""
    if HAS_COACH:
        try:
            return CommunicationCoach()
        except Exception as e:
            return None
    return None

@st.cache_resource
def get_response_generator():
    """Initialize response generator (cached)"""
    if HAS_RESPONSE_GEN:
        try:
            return ResponseGenerator(OPENAI_API_KEY)
        except Exception as e:
            return None
    return None

# Helper function to load JSON from GitHub
def load_from_github(repo_owner, repo_name, file_path, token):
    """Load JSON file from private GitHub repository"""
    try:
        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{file_path}"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

# Load codebooks
@st.cache_data
def load_codebooks():
    """Load codebook data from GitHub, Streamlit secrets, or local files"""
    eppc = None
    sdoh = None
    
    # Try loading from GitHub first (for deployment with private repo)
    try:
        github_token = st.secrets.get('GITHUB_TOKEN', '')
        github_owner = st.secrets.get('GITHUB_REPO_OWNER', 'Elyasirankhah')
        github_repo = st.secrets.get('GITHUB_REPO_NAME', 'JSON_DATA_VOICE_AI_AGENT')
        
        if github_token:
            eppc = load_from_github(github_owner, github_repo, 'codebook_eppc.json', github_token)
            sdoh = load_from_github(github_owner, github_repo, 'codebook_sdoh.json', github_token)
            if eppc and sdoh:
                return eppc, sdoh
    except Exception:
        pass
    
    # Try loading from Streamlit secrets (fallback)
    try:
        if 'codebook_eppc' in st.secrets:
            eppc_str = st.secrets['codebook_eppc']
            eppc = json.loads(eppc_str) if isinstance(eppc_str, str) else eppc_str
        if 'codebook_sdoh' in st.secrets:
            sdoh_str = st.secrets['codebook_sdoh']
            sdoh = json.loads(sdoh_str) if isinstance(sdoh_str, str) else sdoh_str
        if eppc and sdoh:
            return eppc, sdoh
    except Exception:
        pass
    
    # Fall back to local files (for local development)
    try:
        with open('data/codebook_eppc.json', 'r', encoding='utf-8') as f:
            eppc = json.load(f)
        with open('data/codebook_sdoh.json', 'r', encoding='utf-8') as f:
            sdoh = json.load(f)
        return eppc, sdoh
    except FileNotFoundError:
        st.error("""
        ⚠️ Codebooks not found! 
        
        For Streamlit Cloud deployment:
        1. Go to your app settings → Secrets
        2. Add:
           - GITHUB_TOKEN = "your-github-token"
           - GITHUB_REPO_OWNER = "Elyasirankhah"
           - GITHUB_REPO_NAME = "JSON_DATA_VOICE_AI_AGENT"
        
        For local development:
        - Ensure data/codebook_eppc.json and data/codebook_sdoh.json exist
        """)
        return None, None

def generate_narrative(history):
    """Generate a narrative summary from analysis history"""
    if not history:
        return "No analysis history available."
    
    # Count categories
    eppc_counter = Counter()
    sdoh_counter = Counter()
    total_analyses = len(history)
    
    for result in history:
        for item in result.get('eppc', []):
            cat = item.get('category', 'Unknown')
            if cat:
                eppc_counter[cat] += 1
        for item in result.get('sdoh', []):
            cat = item.get('category', 'Unknown')
            if cat:
                sdoh_counter[cat] += 1
    
    # Build narrative
    narrative_parts = []
    narrative_parts.append(f"**Analysis Summary:** Based on {total_analyses} communication analysis{'es' if total_analyses > 1 else ''}, ")
    
    if eppc_counter:
        top_eppc = eppc_counter.most_common(3)
        eppc_list = [f"{cat} ({count}x)" for cat, count in top_eppc]
        narrative_parts.append(f"the most common communication patterns are: {', '.join(eppc_list)}. ")
    else:
        narrative_parts.append("no specific communication patterns were consistently detected. ")
    
    if sdoh_counter:
        top_sdoh = sdoh_counter.most_common(3)
        sdoh_list = [f"{cat} ({count}x)" for cat, count in top_sdoh]
        narrative_parts.append(f"Social determinants of health concerns include: {', '.join(sdoh_list)}. ")
    else:
        narrative_parts.append("no social determinants of health needs were identified. ")
    
    # Add insights
    if total_analyses >= 3:
        narrative_parts.append(f"\n\n**Insights:** This patient has shown consistent communication patterns across {total_analyses} interactions. ")
        if eppc_counter:
            narrative_parts.append("The communication style suggests active engagement with healthcare providers. ")
        if sdoh_counter:
            narrative_parts.append("Social needs have been identified and may require additional support or resources. ")
    
    return "".join(narrative_parts)

def create_timeline_chart(history):
    """Create a timeline visualization of analyses"""
    if len(history) < 2:
        return None
    
    # Prepare data
    timeline_data = []
    for result in history:
        timestamp = datetime.fromisoformat(result['timestamp'])
        eppc_count = len(result.get('eppc', []))
        sdoh_count = len(result.get('sdoh', []))
        
        timeline_data.append({
            'timestamp': timestamp,
            'EPPC Patterns': eppc_count,
            'SDoH Needs': sdoh_count,
            'Total Detections': eppc_count + sdoh_count
        })
    
    df = pd.DataFrame(timeline_data)
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['EPPC Patterns'],
        mode='lines+markers',
        name='EPPC Patterns',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['SDoH Needs'],
        mode='lines+markers',
        name='SDoH Needs',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Communication Patterns Timeline",
        xaxis_title="Date & Time",
        yaxis_title="Number of Detections",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_category_chart(history, category_type, title):
    """Create a bar chart of category frequencies"""
    counter = Counter()
    
    for result in history:
        for item in result.get(category_type, []):
            cat = item.get('category', 'Unknown')
            if cat:
                counter[cat] += 1
    
    if not counter:
        return None
    
    categories = list(counter.keys())
    counts = list(counter.values())
    
    # Color scheme
    colors = px.colors.qualitative.Set3[:len(categories)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Category",
        yaxis_title="Frequency",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_trends_chart(history):
    """Create a trends chart showing category patterns over time"""
    if len(history) < 2:
        return None
    
    # Collect all unique categories
    all_eppc_cats = set()
    all_sdoh_cats = set()
    
    for result in history:
        for item in result.get('eppc', []):
            all_eppc_cats.add(item.get('category', 'Unknown'))
        for item in result.get('sdoh', []):
            all_sdoh_cats.add(item.get('category', 'Unknown'))
    
    # Build time series data
    timeline_data = defaultdict(lambda: defaultdict(int))
    
    for result in history:
        timestamp = datetime.fromisoformat(result['timestamp'])
        date_str = timestamp.strftime('%Y-%m-%d %H:%M')
        
        for item in result.get('eppc', []):
            cat = item.get('category', 'Unknown')
            if cat:
                timeline_data[f"EPPC: {cat}"][date_str] += 1
        
        for item in result.get('sdoh', []):
            cat = item.get('category', 'Unknown')
            if cat:
                timeline_data[f"SDoH: {cat}"][date_str] += 1
    
    if not timeline_data:
        return None
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    color_idx = 0
    
    for category, time_counts in timeline_data.items():
        sorted_times = sorted(time_counts.keys())
        counts = [time_counts[t] for t in sorted_times]
        
        fig.add_trace(go.Scatter(
            x=sorted_times,
            y=counts,
            mode='lines+markers',
            name=category,
            line=dict(color=colors[color_idx % len(colors)], width=2),
            marker=dict(size=6)
        ))
        color_idx += 1
    
    fig.update_layout(
        title="Category Trends Over Time",
        xaxis_title="Date & Time",
        yaxis_title="Occurrences",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
    )
    
    return fig

def analyze_text(text, analyzer_type="llm", threshold=0.5):
    """
    Analyze text using selected analyzer (LLM, ML, or Trained)
    """
    if analyzer_type == "llm":
        llm_analyzer = get_llm_analyzer()
        if llm_analyzer:
            try:
                return llm_analyzer.analyze(text)
            except Exception as e:
                st.error(f"LLM analysis error: {e}")
                return [], []
        else:
            st.warning("LLM analyzer not available. Falling back to ML.")
            analyzer_type = "ml"
    
    if analyzer_type == "trained":
        trained_analyzer = get_trained_analyzer()
        if trained_analyzer:
            try:
                return trained_analyzer.analyze(text, confidence_threshold=0.3)
            except Exception as e:
                st.error(f"Trained model error: {e}")
                return [], []
        else:
            st.warning("Trained model not available. Falling back to LLM.")
            analyzer_type = "llm"
            return analyze_text(text, analyzer_type, threshold)
    
    if analyzer_type == "ml":
        ml_analyzer = get_ml_analyzer()
        if ml_analyzer:
            try:
                return ml_analyzer.analyze(text, threshold=threshold)
            except Exception as e:
                st.error(f"ML analysis error: {e}")
                import traceback
                traceback.print_exc()
                return [], []
        else:
            st.error("ML analyzer not available. Please install: pip install sentence-transformers")
            return [], []
    
    return [], []

# Main App
def main():
    # Load codebooks
    eppc_codebook, sdoh_codebook = load_codebooks()
    if eppc_codebook is None:
        st.stop()
    
    # Initialize page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # CSS for navigation styling
    st.markdown("""
    <style>
    .nav-button {
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation bar - 4 pages
    st.markdown("<br>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("**Home**", use_container_width=True, type="primary" if st.session_state.current_page == "Home" else "secondary", key="nav_home"):
            st.session_state.current_page = "Home"
            st.rerun()
    with nav_col2:
        if st.button("**Analyze**", use_container_width=True, type="primary" if st.session_state.current_page == "Analyze" else "secondary", key="nav_analyze"):
            st.session_state.current_page = "Analyze"
            st.rerun()
    with nav_col3:
        if st.button("**Insights**", use_container_width=True, type="primary" if st.session_state.current_page == "Insights" else "secondary", key="nav_insights"):
            st.session_state.current_page = "Insights"
            st.rerun()
    with nav_col4:
        if st.button("**Coach**", use_container_width=True, type="primary" if st.session_state.current_page == "Coach" else "secondary", key="nav_coach"):
            st.session_state.current_page = "Coach"
            st.rerun()
    
    st.divider()
    
    # Page routing
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Analyze":
        render_analyze_page(eppc_codebook, sdoh_codebook)
    elif st.session_state.current_page == "Insights":
        render_insights_page()
    elif st.session_state.current_page == "Coach":
        render_coach_page()

def render_home_page():
    """Render the Home page with attractive design"""
    
    # Custom CSS for home page styling
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #6c757d;
        font-size: 0.95rem;
    }
    .code-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .sdoh-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    .stat-card {
        background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        display: inline-block;
        text-decoration: none;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">💬 ECHO AI Agent</div>
        <div class="hero-subtitle">Empathetic Communication & Health Orchestrator</div>
        <p style="font-size: 1.1rem; max-width: 700px; margin: 0 auto; opacity: 0.9;">
            AI-powered platform that helps healthcare providers analyze and improve patient communication 
            by detecting communication patterns and social determinants of health.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats if history exists
    if 'results_history' in st.session_state and st.session_state.results_history:
        history = st.session_state.results_history
        total_eppc = sum(len(r.get('eppc', [])) for r in history)
        total_sdoh = sum(len(r.get('sdoh', [])) for r in history)
        
        st.markdown("### 📊 Your Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(history)}</div>
                <div class="stat-label">Total Analyses</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_eppc}</div>
                <div class="stat-label">EPPC Patterns</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_sdoh}</div>
                <div class="stat-label">SDoH Identified</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Smart Analysis</div>
            <div class="feature-desc">
                Upload text, voice recordings, or JSON conversation threads. 
                Our AI detects communication patterns instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Rich Insights</div>
            <div class="feature-desc">
                Visualize trends, track patterns over time, and understand 
                your communication strengths and areas for improvement.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">AI Coach</div>
            <div class="feature-desc">
                Chat with your personalized AI coach for actionable tips 
                based on your specific communication patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # What We Detect Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📞 Communication Patterns (EPPC)")
        st.markdown("""
        <span class="code-badge">Partnership</span>
        <span class="code-badge">Emotional Support</span>
        <span class="code-badge">Information-Giving</span>
        <span class="code-badge">Information-Seeking</span>
        <span class="code-badge">Shared Decision-Making</span>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.info("Detect how patients and providers communicate, collaborate, and make decisions together.")
    
    with col2:
        st.markdown("### 🏥 Social Determinants (SDoH)")
        st.markdown("""
        <span class="sdoh-badge">Financial Insecurity</span>
        <span class="sdoh-badge">Housing</span>
        <span class="sdoh-badge">Food Insecurity</span>
        <span class="sdoh-badge">Transportation</span>
        <span class="sdoh-badge">Employment</span>
        <span class="sdoh-badge">Social Isolation</span>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.warning("Identify social needs that may impact patient health outcomes and care access.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🚀 Ready to Get Started?")
        st.markdown("Navigate to the **Analyze** page to begin analyzing patient-provider conversations.")
        if st.button("➡️ Go to Analyze", type="primary", use_container_width=True):
            st.session_state.current_page = "Analyze"
            st.rerun()

def render_analyze_page(eppc_codebook, sdoh_codebook):
    """Render the Analyze page with split-screen layout"""
    st.title("📊 Analyze Communication")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Analyzer Settings")
        
        # Show available analyzers
        available_analyzers = ["llm", "ml"]
        analyzer_labels = {
            "llm": "🧠 LLM - Best Overall",
            "ml": "🤖 ML (Embeddings)"
        }
        
        if HAS_TRAINED:
            available_analyzers.insert(1, "trained")
            analyzer_labels["trained"] = "🎓 Trained Model (Bio_ClinicalBERT)"
        
        analyzer_type = st.radio(
            "Analysis Method:",
            available_analyzers,
            format_func=lambda x: analyzer_labels[x],
            index=0,
            key="analyzer_selection_analyze"
        )
        
        threshold = 0.6
        
        st.divider()
        
        st.header("📁 Upload Conversation Thread")
        uploaded_file = st.file_uploader(
            "Upload JSON file with conversation thread",
            type=['json'],
            help="Upload a JSON file containing patient-provider conversation messages"
        )
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                # Parse JSON and convert to thread format
                thread_from_json = ""
                processed = False
                
                if isinstance(json_data, dict):
                    # Try 'conversations' array (multiple patients format)
                    if 'conversations' in json_data:
                        conversations = json_data.get('conversations', [])
                        if conversations:
                            thread_from_json = ""
                            total_messages = 0
                            for conv in conversations:
                                patient_id = conv.get('patient_id', '')
                                patient_name = conv.get('patient_name', '')
                                if patient_id or patient_name:
                                    identifier = patient_name if patient_name else f"Patient {patient_id}"
                                    thread_from_json += f"\n--- Conversation with {identifier} ---\n"
                                conv_messages = conv.get('messages', [])
                                for msg in conv_messages:
                                    role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                                    content = msg.get('content', msg.get('message', msg.get('text', '')))
                                    if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                                        thread_from_json += f"Provider: {content}\n"
                                    elif role == 'patient' or role == 'p' or role.startswith('pat'):
                                        thread_from_json += f"Patient: {content}\n"
                                    else:
                                        thread_from_json += f"{role.capitalize()}: {content}\n"
                                    total_messages += 1
                            st.success(f"✅ Successfully loaded {len(conversations)} conversation(s) with {total_messages} total messages")
                            st.session_state.thread_from_json = thread_from_json
                            processed = True
                    
                    # Try 'messages' or 'conversation' array (single conversation format)
                    if not processed:
                        messages = json_data.get('messages', json_data.get('conversation', []))
                        if messages:
                            thread_from_json = ""
                            for msg in messages:
                                role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                                content = msg.get('content', msg.get('message', msg.get('text', '')))
                                if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                                    thread_from_json += f"Provider: {content}\n"
                                elif role == 'patient' or role == 'p' or role.startswith('pat'):
                                    thread_from_json += f"Patient: {content}\n"
                                else:
                                    thread_from_json += f"{role.capitalize()}: {content}\n"
                            st.success(f"✅ Successfully loaded {len(messages)} messages")
                            st.session_state.thread_from_json = thread_from_json
                            processed = True
                
                elif isinstance(json_data, list):
                    # Direct list of messages
                    thread_from_json = ""
                    for msg in json_data:
                        role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                        content = msg.get('content', msg.get('message', msg.get('text', '')))
                        if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                            thread_from_json += f"Provider: {content}\n"
                        elif role == 'patient' or role == 'p' or role.startswith('pat'):
                            thread_from_json += f"Patient: {content}\n"
                        else:
                            thread_from_json += f"{role.capitalize()}: {content}\n"
                    st.success(f"✅ Successfully loaded {len(json_data)} messages")
                    st.session_state.thread_from_json = thread_from_json
                    processed = True
                
                if processed and 'thread_from_json' in st.session_state:
                    if st.button("🔍 Analyze Thread", type="primary", use_container_width=True):
                        thread_analyzer = get_thread_analyzer()
                        if thread_analyzer:
                            with st.spinner("Analyzing conversation thread..."):
                                thread_analysis = thread_analyzer.analyze_thread(st.session_state.thread_from_json)
                                
                                # Store in history
                                if 'results_history' not in st.session_state:
                                    st.session_state.results_history = []
                                
                                result = {
                                    'timestamp': datetime.now().isoformat(),
                                    'text': st.session_state.thread_from_json[:500] + "...",
                                    'type': 'thread',
                                    'thread_analysis': thread_analysis,
                                    'eppc': thread_analysis.get('patient_communication', {}).get('eppc_codes', []),
                                    'sdoh': thread_analysis.get('patient_communication', {}).get('sdoh_codes', [])
                                }
                                st.session_state.results_history.append(result)
                                st.session_state.last_thread_analysis = thread_analysis
                                st.success("✅ Thread analysis complete! Check Insights page for details.")
                                st.rerun()
                        else:
                            st.error("Thread analyzer not available")
                
            except Exception as e:
                st.error(f"Error parsing JSON: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    # Main content - Use session state for tab control
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "text"  # Default to text
    
    # Tab navigation with buttons - 4 tabs now
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📝 Text Input", use_container_width=True, type="primary" if st.session_state.current_view == "text" else "secondary"):
            st.session_state.current_view = "text"
            st.rerun()
    with col2:
        if st.button("🎤 Voice Input", use_container_width=True, type="primary" if st.session_state.current_view == "voice" else "secondary"):
            st.session_state.current_view = "voice"
            st.rerun()
    with col3:
        if st.button("📁 Thread Upload", use_container_width=True, type="primary" if st.session_state.current_view == "thread" else "secondary"):
            st.session_state.current_view = "thread"
            st.rerun()
    with col4:
        if st.button("📊 Results History", use_container_width=True, type="primary" if st.session_state.current_view == "history" else "secondary"):
            st.session_state.current_view = "history"
            st.rerun()
    
    st.divider()
    
    # Show content based on current view
    if st.session_state.current_view == "text":
        st.subheader("Enter Text to Analyze")
        
        text_input = st.text_area(
            "Paste or type patient message:",
            height=150,
            placeholder="Example: I can't afford my medication anymore and I'm worried about my test results..."
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input:
                try:
                    with st.spinner(f"Analyzing with {analyzer_type.upper()}..."):
                        eppc_results, sdoh_results = analyze_text(
                            text_input, analyzer_type=analyzer_type, threshold=threshold
                        )
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    eppc_results, sdoh_results = [], []
                
                # Store in session state
                if 'results_history' not in st.session_state:
                    st.session_state.results_history = []
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'text': text_input,
                    'eppc': eppc_results,
                    'sdoh': sdoh_results
                }
                st.session_state.results_history.append(result)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📞 Communication Patterns (EPPC)")
                    if eppc_results:
                        for item in eppc_results:
                            try:
                                category = item.get('category', 'Unknown')
                                confidence = item.get('confidence', 0.0)
                                # Handle if confidence is string
                                if isinstance(confidence, str):
                                    try:
                                        confidence = float(confidence)
                                    except:
                                        confidence = 0.0
                                matched_phrase = item.get('matched_phrase', item.get('reason', 'N/A'))
                                
                                st.success(f"**{category}** (Confidence: {confidence:.0%})")
                                if matched_phrase and matched_phrase != 'N/A' and len(str(matched_phrase)) > 0:
                                    st.caption(f"Reason: '{matched_phrase[:100]}'")  # Truncate long reasons
                            except Exception as e:
                                st.error(f"Error displaying result: {e}")
                                st.json(item)  # Show raw data for debugging
                    else:
                        st.info("No communication patterns detected")
                
                with col2:
                    st.subheader("🏥 Social Determinants (SDoH)")
                    if sdoh_results:
                        for item in sdoh_results:
                            try:
                                category = item.get('category', 'Unknown')
                                confidence = item.get('confidence', 0.0)
                                # Handle if confidence is string
                                if isinstance(confidence, str):
                                    try:
                                        confidence = float(confidence)
                                    except:
                                        confidence = 0.0
                                matched_phrase = item.get('matched_phrase', item.get('reason', 'N/A'))
                                
                                st.warning(f"**{category}** (Confidence: {confidence:.0%})")
                                if matched_phrase and matched_phrase != 'N/A' and len(str(matched_phrase)) > 0:
                                    st.caption(f"Reason: '{matched_phrase[:100]}'")  # Truncate long reasons
                            except Exception as e:
                                st.error(f"Error displaying result: {e}")
                                st.json(item)  # Show raw data for debugging
                    else:
                        st.info("No SDoH needs detected")
            else:
                st.warning("Please enter some text to analyze")
    
    elif st.session_state.current_view == "voice":
        st.subheader("🎤 Voice Input")
        
        if not HAS_VOICE:
            st.warning("⚠️ Voice transcription not available. Install dependencies: pip install openai")
            st.info("For now, use Text Input tab.")
        else:
            st.markdown("**Record or upload audio to transcribe and analyze**")
            
            # Audio input options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎙️ Option 1: Live Recording")
                st.markdown("**Click the microphone to record live from your device**")
                
                # Reset audio inputs if processing is complete
                if st.session_state.get('audio_processed', False):
                    st.session_state.audio_processed = False
                
                live_audio = st.audio_input(
                    "Record your voice",
                    help="Click to start recording. Your browser will ask for microphone permission.",
                    key="live_audio_input"
                )
                
                if live_audio is not None:
                    st.audio(live_audio, format="audio/wav")
                    st.success("✅ Recording complete! Click 'Transcribe & Analyze' below.")
            
            with col2:
                st.markdown("### 📁 Option 2: Upload Audio File")
                audio_file = st.file_uploader(
                    "Upload audio file",
                    type=['wav', 'mp3', 'ogg', 'webm', 'm4a'],
                    help="Supported formats: WAV, MP3, OGG, WebM, M4A"
                )
                
                if audio_file is not None:
                    st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
            
            # Determine which audio source to use
            audio_to_process = None
            audio_format = "wav"
            
            if live_audio is not None:
                audio_to_process = live_audio
                audio_format = "wav"
            elif audio_file is not None:
                audio_to_process = audio_file
                audio_format = audio_file.name.split('.')[-1].lower()
            
            # Process audio if available
            if audio_to_process is not None:
                st.divider()
                
                # Transcribe button
                if st.button("🎙️ Transcribe & Analyze", type="primary", use_container_width=True):
                    with st.spinner("Transcribing audio... This may take a few seconds."):
                        try:
                            # Read audio bytes
                            if isinstance(audio_to_process, bytes):
                                # Live recording (already bytes)
                                audio_bytes = audio_to_process
                            else:
                                # File upload (need to read)
                                audio_bytes = audio_to_process.read()
                            
                            # Use detected format
                            file_format = audio_format
                            
                            # Transcribe (force English language)
                            transcribed_text = transcribe_audio_bytes(
                                audio_bytes,
                                input_format=file_format,
                                language="en",  # Force English transcription
                                api_key=OPENAI_API_KEY
                            )
                            
                            # Store in session state (but don't show it yet)
                            st.session_state.transcribed_text = transcribed_text
                            st.session_state.audio_transcribed = True
                            
                            st.success("✅ Transcription complete! Click 'Analyze' below to see results.")
                            
                            # Don't rerun here - let user click Analyze button
                            
                        except Exception as e:
                            st.error(f"❌ Transcription failed: {str(e)}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
                
                # Show Analyze button if transcription is complete but not analyzed yet
                if st.session_state.get('audio_transcribed', False) and 'transcribed_text' in st.session_state and st.session_state.transcribed_text:
                    if not st.session_state.get('voice_analyzed', False):
                        st.divider()
                        if st.button("🔍 Analyze Transcribed Text", type="primary", use_container_width=True):
                            transcribed_text = st.session_state.transcribed_text
                            
                            if transcribed_text.strip():
                                eppc_results, sdoh_results = [], []
                                try:
                                    with st.spinner(f"Analyzing with {analyzer_type.upper()}..."):
                                        eppc_results, sdoh_results = analyze_text(
                                            transcribed_text, analyzer_type=analyzer_type, threshold=threshold
                                        )
                                except Exception as e:
                                    st.error(f"Analysis error: {str(e)}")
                                    import traceback
                                    with st.expander("Error Details"):
                                        st.code(traceback.format_exc())
                                    eppc_results, sdoh_results = [], []
                                
                                # Store in session state
                                if 'results_history' not in st.session_state:
                                    st.session_state.results_history = []
                                
                                result = {
                                    'timestamp': datetime.now().isoformat(),
                                    'text': transcribed_text,
                                    'eppc': eppc_results,
                                    'sdoh': sdoh_results
                                }
                                st.session_state.results_history.append(result)
                                
                                # Store results for display
                                st.session_state.last_eppc_results = eppc_results
                                st.session_state.last_sdoh_results = sdoh_results
                                st.session_state.voice_analyzed = True
                                st.session_state.active_tab = 1  # Stay on voice tab
                    
                    # Show results if analyzed
                    if st.session_state.get('voice_analyzed', False):
                        st.divider()
                        st.markdown("### 📊 Analysis Results")
                        
                        # Show the transcribed sentence
                        st.markdown("**📝 Transcribed Text:**")
                        st.info(f'"{st.session_state.transcribed_text}"')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📞 Communication Patterns (EPPC)")
                            eppc_results = st.session_state.get('last_eppc_results', [])
                            if eppc_results:
                                for item in eppc_results:
                                    try:
                                        category = item.get('category', 'Unknown')
                                        confidence = item.get('confidence', 0.0)
                                        if isinstance(confidence, str):
                                            try:
                                                confidence = float(confidence)
                                            except:
                                                confidence = 0.0
                                        matched_phrase = item.get('matched_phrase', item.get('reason', 'N/A'))
                                        
                                        st.success(f"**{category}** (Confidence: {confidence:.0%})")
                                        if matched_phrase and matched_phrase != 'N/A' and len(str(matched_phrase)) > 0:
                                            st.caption(f"Reason: '{str(matched_phrase)[:100]}'")
                                    except Exception as e:
                                        st.error(f"Error displaying result: {e}")
                                        st.json(item)
                            else:
                                st.info("No communication patterns detected")
                        
                        with col2:
                            st.subheader("🏥 Social Determinants (SDoH)")
                            sdoh_results = st.session_state.get('last_sdoh_results', [])
                            if sdoh_results:
                                for item in sdoh_results:
                                    try:
                                        category = item.get('category', 'Unknown')
                                        confidence = item.get('confidence', 0.0)
                                        if isinstance(confidence, str):
                                            try:
                                                confidence = float(confidence)
                                            except:
                                                confidence = 0.0
                                        matched_phrase = item.get('matched_phrase', item.get('reason', 'N/A'))
                                        
                                        st.warning(f"**{category}** (Confidence: {confidence:.0%})")
                                        if matched_phrase and matched_phrase != 'N/A' and len(str(matched_phrase)) > 0:
                                            st.caption(f"Reason: '{str(matched_phrase)[:100]}'")
                                    except Exception as e:
                                        st.error(f"Error displaying result: {e}")
                                        st.json(item)
                            else:
                                st.info("No SDoH needs detected")
                        
                        # Reset button to start over
                        if st.button("🔄 Record Again", type="secondary"):
                            # Clear all voice-related state
                            if 'transcribed_text' in st.session_state:
                                del st.session_state.transcribed_text
                            if 'audio_transcribed' in st.session_state:
                                del st.session_state.audio_transcribed
                            if 'voice_analyzed' in st.session_state:
                                del st.session_state.voice_analyzed
                            if 'last_eppc_results' in st.session_state:
                                del st.session_state.last_eppc_results
                            if 'last_sdoh_results' in st.session_state:
                                del st.session_state.last_sdoh_results
                            st.rerun()
    
    elif st.session_state.current_view == "thread":
        st.subheader("📁 Upload Conversation Thread")
        st.markdown("Upload a JSON file containing patient-provider conversation messages to analyze the full thread.")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload a JSON file containing conversation messages"
        )
        
        if uploaded_file is not None:
            try:
                json_data = json.load(uploaded_file)
                # Parse JSON and convert to thread format
                thread_from_json = ""
                processed = False
                
                if isinstance(json_data, dict):
                    # Try 'conversations' array (multiple patients format)
                    if 'conversations' in json_data:
                        conversations = json_data.get('conversations', [])
                        if conversations:
                            thread_from_json = ""
                            total_messages = 0
                            for conv in conversations:
                                patient_id = conv.get('patient_id', '')
                                patient_name = conv.get('patient_name', '')
                                if patient_id or patient_name:
                                    identifier = patient_name if patient_name else f"Patient {patient_id}"
                                    thread_from_json += f"\n--- Conversation with {identifier} ---\n"
                                conv_messages = conv.get('messages', [])
                                for msg in conv_messages:
                                    role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                                    content = msg.get('content', msg.get('message', msg.get('text', '')))
                                    if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                                        thread_from_json += f"Provider: {content}\n"
                                    elif role == 'patient' or role == 'p' or role.startswith('pat'):
                                        thread_from_json += f"Patient: {content}\n"
                                    else:
                                        thread_from_json += f"{role.capitalize()}: {content}\n"
                                    total_messages += 1
                            st.success(f"✅ Successfully loaded {len(conversations)} conversation(s) with {total_messages} total messages")
                            st.session_state.thread_from_json = thread_from_json
                            processed = True
                    
                    # Try 'messages' or 'conversation' array (single conversation format)
                    if not processed:
                        messages = json_data.get('messages', json_data.get('conversation', []))
                        if messages:
                            thread_from_json = ""
                            for msg in messages:
                                role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                                content = msg.get('content', msg.get('message', msg.get('text', '')))
                                if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                                    thread_from_json += f"Provider: {content}\n"
                                elif role == 'patient' or role == 'p' or role.startswith('pat'):
                                    thread_from_json += f"Patient: {content}\n"
                                else:
                                    thread_from_json += f"{role.capitalize()}: {content}\n"
                            st.success(f"✅ Successfully loaded {len(messages)} messages")
                            st.session_state.thread_from_json = thread_from_json
                            processed = True
                
                elif isinstance(json_data, list):
                    # Direct list of messages
                    thread_from_json = ""
                    for msg in json_data:
                        role = msg.get('role', msg.get('sender', 'Unknown')).lower().strip()
                        content = msg.get('content', msg.get('message', msg.get('text', '')))
                        if role == 'provider' or role == 'doctor' or role == 'clinician' or role == 'dr' or role.startswith('prov'):
                            thread_from_json += f"Provider: {content}\n"
                        elif role == 'patient' or role == 'p' or role.startswith('pat'):
                            thread_from_json += f"Patient: {content}\n"
                        else:
                            thread_from_json += f"{role.capitalize()}: {content}\n"
                    st.success(f"✅ Successfully loaded {len(json_data)} messages")
                    st.session_state.thread_from_json = thread_from_json
                    processed = True
                
                # Show preview
                if processed and 'thread_from_json' in st.session_state:
                    st.markdown("### Preview")
                    with st.expander("View parsed conversation", expanded=False):
                        st.text(st.session_state.thread_from_json[:2000] + ("..." if len(st.session_state.thread_from_json) > 2000 else ""))
                    
                    if st.button("🔍 Analyze Thread", type="primary", use_container_width=True):
                        thread_analyzer = get_thread_analyzer()
                        all_eppc = []
                        all_sdoh = []
                        thread_analysis = None
                        
                        if thread_analyzer:
                            with st.spinner("Analyzing conversation thread..."):
                                thread_analysis = thread_analyzer.analyze_thread(st.session_state.thread_from_json)
                                all_eppc = thread_analysis.get('patient_communication', {}).get('eppc_codes', [])
                                all_sdoh = thread_analysis.get('patient_communication', {}).get('sdoh_codes', [])
                        else:
                            # Fallback: analyze each message individually
                            with st.spinner("Analyzing messages individually..."):
                                lines = st.session_state.thread_from_json.strip().split('\n')
                                for line in lines:
                                    if line.startswith('Patient:'):
                                        msg_text = line.replace('Patient:', '').strip()
                                        if msg_text:
                                            eppc_results, sdoh_results = analyze_text(msg_text, analyzer_type="llm", threshold=0.5)
                                            all_eppc.extend(eppc_results)
                                            all_sdoh.extend(sdoh_results)
                        
                        # Store in history
                        if 'results_history' not in st.session_state:
                            st.session_state.results_history = []
                        
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'text': st.session_state.thread_from_json[:500] + "...",
                            'type': 'thread',
                            'thread_analysis': thread_analysis,
                            'eppc': all_eppc,
                            'sdoh': all_sdoh
                        }
                        st.session_state.results_history.append(result)
                        st.session_state.last_thread_result = result
                        st.session_state.thread_analyzed = True
                        st.rerun()
                
                # Show results if analyzed
                if st.session_state.get('thread_analyzed', False) and 'last_thread_result' in st.session_state:
                    result = st.session_state.last_thread_result
                    all_eppc = result.get('eppc', [])
                    all_sdoh = result.get('sdoh', [])
                    
                    st.markdown("---")
                    st.markdown("### 📊 Thread Analysis Results")
                    st.success(f"✅ Analysis complete! Found **{len(all_eppc)}** EPPC codes and **{len(all_sdoh)}** SDoH codes.")
                    
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📞 Communication Patterns (EPPC)")
                        if all_eppc:
                            # Count by category
                            eppc_counts = {}
                            for item in all_eppc:
                                cat = item.get('category', 'Unknown')
                                eppc_counts[cat] = eppc_counts.get(cat, 0) + 1
                            
                            for cat, count in sorted(eppc_counts.items(), key=lambda x: x[1], reverse=True):
                                st.success(f"**{cat}** ({count}x)")
                        else:
                            st.info("No communication patterns detected")
                    
                    with col2:
                        st.subheader("🏥 Social Determinants (SDoH)")
                        if all_sdoh:
                            # Count by category
                            sdoh_counts = {}
                            for item in all_sdoh:
                                cat = item.get('category', 'Unknown')
                                sdoh_counts[cat] = sdoh_counts.get(cat, 0) + 1
                            
                            for cat, count in sorted(sdoh_counts.items(), key=lambda x: x[1], reverse=True):
                                st.warning(f"**{cat}** ({count}x)")
                        else:
                            st.info("No SDoH needs detected")
                    
                    # Show detailed results
                    with st.expander("📋 View Detailed Results"):
                        if all_eppc:
                            st.markdown("**EPPC Details:**")
                            for i, item in enumerate(all_eppc, 1):
                                category = item.get('category', 'Unknown')
                                confidence = item.get('confidence', 0.0)
                                if isinstance(confidence, str):
                                    try:
                                        confidence = float(confidence)
                                    except:
                                        confidence = 0.0
                                reason = item.get('matched_phrase', item.get('reason', 'N/A'))
                                st.markdown(f"{i}. **{category}** ({confidence:.0%}) - {str(reason)[:100]}")
                        
                        if all_sdoh:
                            st.markdown("**SDoH Details:**")
                            for i, item in enumerate(all_sdoh, 1):
                                category = item.get('category', 'Unknown')
                                confidence = item.get('confidence', 0.0)
                                if isinstance(confidence, str):
                                    try:
                                        confidence = float(confidence)
                                    except:
                                        confidence = 0.0
                                reason = item.get('matched_phrase', item.get('reason', 'N/A'))
                                st.markdown(f"{i}. **{category}** ({confidence:.0%}) - {str(reason)[:100]}")
                    
                    st.info("💡 Results have been saved to **Insights** and **Results History** pages.")
                    
                    # Reset button
                    if st.button("🔄 Analyze Another Thread", type="secondary"):
                        if 'thread_analyzed' in st.session_state:
                            del st.session_state.thread_analyzed
                        if 'last_thread_result' in st.session_state:
                            del st.session_state.last_thread_result
                        if 'thread_from_json' in st.session_state:
                            del st.session_state.thread_from_json
                        st.rerun()
                
            except Exception as e:
                st.error(f"Error parsing JSON: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        # Show sample JSON format
        st.markdown("---")
        st.markdown("### Sample JSON Format")
        st.code('''[
  {
    "role": "patient",
    "content": "I'm worried about my test results..."
  },
  {
    "role": "provider", 
    "content": "I understand your concern..."
  }
]''', language="json")
    
    elif st.session_state.current_view == "history":
        st.subheader("📊 Analysis History & Narrative Dashboard")
        
        if 'results_history' in st.session_state and st.session_state.results_history:
            history = st.session_state.results_history
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.results_history = []
                st.rerun()
            
            # Narrative Summary Section
            st.markdown("---")
            st.markdown("### 📖 Communication Narrative Summary")
            
            # Generate narrative
            narrative = generate_narrative(history)
            st.info(narrative)
            
            # Visualizations Section
            st.markdown("---")
            st.markdown("### 📈 Visual Analytics")
            
            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "📊 Timeline", 
                "📋 Category Frequency", 
                "🎯 Pattern Trends",
                "📝 Detailed History"
            ])
            
            with viz_tab1:
                st.markdown("#### Timeline of Communication Patterns")
                timeline_fig = create_timeline_chart(history)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.info("Need at least 2 analyses to show timeline")
            
            with viz_tab2:
                st.markdown("#### Category Frequency Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    eppc_fig = create_category_chart(history, 'eppc', 'Communication Patterns (EPPC)')
                    if eppc_fig:
                        st.plotly_chart(eppc_fig, use_container_width=True)
                    else:
                        st.info("No EPPC patterns detected yet")
                
                with col2:
                    sdoh_fig = create_category_chart(history, 'sdoh', 'Social Determinants (SDoH)')
                    if sdoh_fig:
                        st.plotly_chart(sdoh_fig, use_container_width=True)
                    else:
                        st.info("No SDoH needs detected yet")
            
            with viz_tab3:
                st.markdown("#### Pattern Trends Over Time")
                trends_fig = create_trends_chart(history)
                if trends_fig:
                    st.plotly_chart(trends_fig, use_container_width=True)
                else:
                    st.info("Need multiple analyses to show trends")
            
            with viz_tab4:
                st.markdown("#### Detailed Analysis History")
                for i, result in enumerate(reversed(history[-10:]), 1):
                    result_id = len(history) - i + 1
                    with st.expander(f"Analysis #{result_id} - {result['timestamp'][:19]}"):
                        st.text_area(
                            "Original Text", 
                            result['text'], 
                            height=100, 
                            disabled=True,
                            key=f"history_text_{result_id}_{result['timestamp']}"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            eppc_cats = [r.get('category', 'Unknown') for r in result['eppc']]
                            st.write("**EPPC:**", ", ".join(eppc_cats) if eppc_cats else "None")
                        with col2:
                            sdoh_cats = [r.get('category', 'Unknown') for r in result['sdoh']]
                            st.write("**SDoH:**", ", ".join(sdoh_cats) if sdoh_cats else "None")
        else:
            st.info("No analysis history yet. Run some analyses first!")
            st.markdown("""
            **💡 Tip:** After analyzing text or voice input, your results will appear here with:
            - 📖 Narrative summaries of communication patterns
            - 📈 Timeline visualizations
            - 📊 Category frequency charts
            - 🎯 Trend analysis over time
            """)

def render_insights_page():
    """Render the Insights page with analytics"""
    st.title("📈 Insights & Analytics")
    
    if 'results_history' not in st.session_state or not st.session_state.results_history:
        st.info("No analysis history yet. Run some analyses on the Analyze page first!")
        return
    
    history = st.session_state.results_history
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.results_history = []
        st.rerun()
    
    # Narrative Summary Section
    st.markdown("---")
    st.markdown("### 📖 Communication Narrative Summary")
    
    # Generate narrative
    narrative = generate_narrative(history)
    st.info(narrative)
    
    # Visualizations Section
    st.markdown("---")
    st.markdown("### 📈 Visual Analytics")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📊 Timeline", 
        "📋 Category Frequency", 
        "🎯 Pattern Trends",
        "📝 Detailed History"
    ])
    
    with viz_tab1:
        st.markdown("#### Timeline of Communication Patterns")
        timeline_fig = create_timeline_chart(history)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Need at least 2 analyses to show timeline")
    
    with viz_tab2:
        st.markdown("#### Category Frequency Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            eppc_fig = create_category_chart(history, 'eppc', 'Communication Patterns (EPPC)')
            if eppc_fig:
                st.plotly_chart(eppc_fig, use_container_width=True)
            else:
                st.info("No EPPC patterns detected yet")
        
        with col2:
            sdoh_fig = create_category_chart(history, 'sdoh', 'Social Determinants (SDoH)')
            if sdoh_fig:
                st.plotly_chart(sdoh_fig, use_container_width=True)
            else:
                st.info("No SDoH needs detected yet")
    
    with viz_tab3:
        st.markdown("#### Pattern Trends Over Time")
        trends_fig = create_trends_chart(history)
        if trends_fig:
            st.plotly_chart(trends_fig, use_container_width=True)
        else:
            st.info("Need multiple analyses to show trends")
    
    with viz_tab4:
        st.markdown("#### Detailed Analysis History")
        for i, result in enumerate(reversed(history[-10:]), 1):
            result_id = len(history) - i + 1
            with st.expander(f"Analysis #{result_id} - {result['timestamp'][:19]}"):
                st.text_area(
                    "Original Text", 
                    result['text'], 
                    height=100, 
                    disabled=True,
                    key=f"history_text_{result_id}_{result['timestamp']}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    eppc_cats = [r.get('category', 'Unknown') for r in result['eppc']]
                    st.write("**EPPC:**", ", ".join(eppc_cats) if eppc_cats else "None")
                with col2:
                    sdoh_cats = [r.get('category', 'Unknown') for r in result['sdoh']]
                    st.write("**SDoH:**", ", ".join(sdoh_cats) if sdoh_cats else "None")

def render_coach_page():
    """Render the Coach page with chat-based AI coaching"""
    st.title("💬 Communication Coach")
    
    # Initialize chat session
    if 'coach_chat_session' not in st.session_state:
        st.session_state.coach_chat_session = {'messages': []}
    
    chat_session = st.session_state.coach_chat_session
    
    # Get analysis history for context
    history = st.session_state.get('results_history', [])
    
    # Get coach
    coach = get_communication_coach()
    if coach and history:
        coaching_summary = coach.generate_coaching_summary(history)
        
        st.markdown("### 📋 Your Coaching Summary")
        st.info(coaching_summary['summary'])
        
        if coaching_summary['recommendations']:
            st.markdown("### 💡 Recommendations")
            for rec in coaching_summary['recommendations']:
                with st.expander(f"**{rec['pattern']}** (detected {rec['frequency']} times)"):
                    for tip in rec['tips']:
                        st.markdown(f"- {tip}")
    
    st.markdown("---")
    st.markdown("### 💬 Chat with Your AI Coach")
    
    # Show what data the AI has access to - per conversation
    if history:
        st.markdown(f"### 📋 AI Coach knows about your {len(history)} conversation(s):")
        
        for i, result in enumerate(history, 1):
            if result.get('type') == 'thread':
                thread_analysis = result.get('thread_analysis', {})
                patient_comm = thread_analysis.get('patient_communication', {})
                eppc_codes = patient_comm.get('eppc_codes', [])
                sdoh_codes = patient_comm.get('sdoh_codes', [])
            else:
                eppc_codes = result.get('eppc', [])
                sdoh_codes = result.get('sdoh', [])
            
            eppc_cats = list(set([r.get('category', 'Unknown') for r in eppc_codes]))
            sdoh_cats = list(set([r.get('category', 'Unknown') for r in sdoh_codes]))
            
            # Get preview of text
            text_preview = result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', '')
            
            with st.expander(f"**Conversation #{i}** - {result.get('timestamp', '')[:16]}"):
                st.markdown(f"**Preview**: _{text_preview}_")
                if eppc_cats:
                    st.markdown(f"**EPPC**: {', '.join(eppc_cats)}")
                if sdoh_cats:
                    st.markdown(f"**SDoH**: {', '.join(sdoh_cats)}")
        
        st.info("💡 Ask about specific conversations by number (e.g., 'What about conversation #1?') or ask general questions!")
    else:
        st.warning("⚠️ No analysis history yet. Analyze some messages first to get personalized coaching based on your patterns.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in chat_session['messages']:
            role = msg.get('role', 'user')
            message = msg.get('message', '')
            if role == 'user':
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant"):
                    st.write(message)
    
    # Chat input form
    with st.form(key="coach_chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([10, 1])
        with col_input:
            user_input = st.text_input(
                "Type your question or instruction...",
                key="coach_chat_input",
                placeholder="e.g., 'How can I improve my empathy?'",
                label_visibility="collapsed"
            )
        with col_send:
            send_button = st.form_submit_button("➤", type="primary", use_container_width=True)
        
        if send_button and user_input and user_input.strip():
            user_msg = user_input.strip()
            chat_session['messages'].append({
                'role': 'user',
                'message': user_msg,
                'timestamp': None
            })
            
            with st.spinner("🤔 Thinking..."):
                try:
                    if HAS_CHAT_AGENT and OPENAI_API_KEY:
                        chat_agent = get_chat_agent()
                        if chat_agent:
                            # Build detailed per-conversation context
                            conversations_detail = []
                            all_eppc_codes = []
                            all_sdoh_codes = []
                            
                            for i, result in enumerate(history, 1):
                                if result.get('type') == 'thread':
                                    thread_analysis = result.get('thread_analysis', {})
                                    patient_comm = thread_analysis.get('patient_communication', {})
                                    eppc_codes = patient_comm.get('eppc_codes', [])
                                    sdoh_codes = patient_comm.get('sdoh_codes', [])
                                else:
                                    eppc_codes = result.get('eppc', [])
                                    sdoh_codes = result.get('sdoh', [])
                                
                                eppc_cats = list(set([r.get('category', 'Unknown') for r in eppc_codes]))
                                sdoh_cats = list(set([r.get('category', 'Unknown') for r in sdoh_codes]))
                                text_preview = result.get('text', '')[:300]
                                
                                conversations_detail.append({
                                    'number': i,
                                    'timestamp': result.get('timestamp', '')[:16],
                                    'text_preview': text_preview,
                                    'eppc': eppc_cats,
                                    'sdoh': sdoh_cats
                                })
                                
                                all_eppc_codes.extend(eppc_codes)
                                all_sdoh_codes.extend(sdoh_codes)
                            
                            coaching_context = {
                                'conversations': conversations_detail,
                                'eppc_codes': [r.get('category', 'Unknown') for r in all_eppc_codes],
                                'sdoh_codes': [r.get('category', 'Unknown') for r in all_sdoh_codes],
                                'total_analyses': len(history),
                                'coaching_mode': True
                            }
                            
                            if chat_agent.client:
                                response = chat_agent.chat(user_msg, chat_session['messages'], coaching_context)
                                chat_session['messages'].append({
                                    'role': 'agent',
                                    'message': response,
                                    'timestamp': None
                                })
                            else:
                                chat_session['messages'].append({
                                    'role': 'agent',
                                    'message': "I'm here to help! Based on your analysis history, I can offer tips. What would you like to discuss?",
                                    'timestamp': None
                                })
                        else:
                            chat_session['messages'].append({
                                'role': 'agent',
                                'message': "I'm here to help! Based on your analysis history, I can offer tips. What would you like to discuss?",
                                'timestamp': None
                            })
                    else:
                        chat_session['messages'].append({
                            'role': 'agent',
                            'message': "Thank you for your question! Review the coaching summary above for insights. For interactive coaching, please ensure OpenAI API key is configured.",
                            'timestamp': None
                        })
                except Exception as e:
                    chat_session['messages'].append({
                        'role': 'agent',
                        'message': f"I apologize, I encountered an error: {str(e)}. Please try rephrasing your question.",
                        'timestamp': None
                    })
            st.rerun()

if __name__ == "__main__":
    main()
