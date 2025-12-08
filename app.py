"""
Voice Agent MVP - Streamlit App
Main application for voice input and text analysis
Uses ML/LLM for accurate analysis
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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in .env file")
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
    page_title="Patient Voice Agent",
    page_icon="🎤",
    layout="wide"
)

# Initialize analyzers
@st.cache_resource
def get_ml_analyzer():
    """Initialize ML analyzer (cached)"""
    if HAS_ML:
        try:
            analyzer = MLAnalyzer('data/codebook_eppc.json', 'data/codebook_sdoh.json')
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
            return LLMAnalyzer('data/codebook_eppc.json', 'data/codebook_sdoh.json', OPENAI_API_KEY)
        except Exception as e:
            st.error(f"Error initializing LLM analyzer: {e}")
            return None
    return None

@st.cache_resource
def get_trained_analyzer():
    """Initialize trained model analyzer (cached)"""
    if HAS_TRAINED:
        try:
            return TrainedAnalyzer('./trained_models/final_model')
        except Exception as e:
            st.error(f"Error initializing trained analyzer: {e}")
            return None
    return None

# Load codebooks
@st.cache_data
def load_codebooks():
    """Load codebook data"""
    try:
        with open('data/codebook_eppc.json', 'r', encoding='utf-8') as f:
            eppc = json.load(f)
        with open('data/codebook_sdoh.json', 'r', encoding='utf-8') as f:
            sdoh = json.load(f)
        return eppc, sdoh
    except FileNotFoundError:
        st.error("Codebooks not found! Run extract_codebook.py first")
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
    st.title("🎤 Patient Voice Agent MVP")
    st.markdown("**Analyze patient-provider communication patterns from voice or text**")
    
    # Load codebooks
    eppc_codebook, sdoh_codebook = load_codebooks()
    if eppc_codebook is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Choose analyzer type
        2. Enter text or use voice input
        3. Click "Analyze"
        4. View detected communication patterns and SDoH needs
        """)
        
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
            key="analyzer_selection"
        )
        
        # Clear results if analyzer changed (but keep the text/voice input)
        if 'previous_analyzer' not in st.session_state:
            st.session_state.previous_analyzer = analyzer_type
        
        if st.session_state.previous_analyzer != analyzer_type:
            # Analyzer changed - clear old results but keep inputs
            st.session_state.previous_analyzer = analyzer_type
            if 'last_eppc_results' in st.session_state:
                del st.session_state.last_eppc_results
            if 'last_sdoh_results' in st.session_state:
                del st.session_state.last_sdoh_results
            if 'voice_analyzed' in st.session_state:
                st.session_state.voice_analyzed = False  # Reset to allow re-analysis
            st.info(f"🔄 Switched to {analyzer_labels[analyzer_type]}. Click 'Analyze' again to see new results.")
        
        # Default threshold for all analyzers
        threshold = 0.6
        
        st.divider()
        
        st.header("🔍 Detection Categories")
        st.markdown("**EPPC Codes:**")
        for cat in eppc_codebook.keys():
            st.markdown(f"- {cat}")
        
        st.markdown("**SDoH Categories:**")
        for cat in sdoh_codebook.keys():
            st.markdown(f"- {cat}")
    
    # Main content - Use session state for tab control
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "voice"  # Default to voice
    
    # Tab navigation with buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 Text Input", use_container_width=True, type="primary" if st.session_state.current_view == "text" else "secondary"):
            st.session_state.current_view = "text"
            st.rerun()
    with col2:
        if st.button("🎤 Voice Input", use_container_width=True, type="primary" if st.session_state.current_view == "voice" else "secondary"):
            st.session_state.current_view = "voice"
            st.rerun()
    with col3:
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

if __name__ == "__main__":
    main()
