# 🎤 Patient Voice Agent MVP

An AI-powered application that analyzes patient-provider communication patterns and social determinants of health from text and voice input. Built with multiple analyzer approaches including LLM, ML embeddings, and fine-tuned transformers.

## 🌟 Features

- **Voice Input**: Real-time voice recording and transcription using OpenAI Whisper
- **Text Analysis**: Direct text input for quick analysis
- **Multiple Analyzers**:
  - 🧠 LLM (GPT-4) - Best overall accuracy (82%)
  - 🤖 ML Embeddings - Fast semantic similarity matching
  - 🎓 Fine-tuned Bio_ClinicalBERT - Domain-specific model
- **Dual Analysis**:
  - 📞 **EPPC (Electronic Patient-Provider Communication)**: Partnership, Emotional Support, Information-Giving, Information-Seeking, Shared Decision-Making
  - 🏥 **SDoH (Social Determinants of Health)**: Housing, Financial, Food, Transportation, Employment
- **Visual Analytics Dashboard**:
  - 📖 Narrative summaries
  - 📈 Timeline visualizations
  - 📊 Category frequency charts
  - 🎯 Pattern trends analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- ffmpeg (for audio processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/patient-voice-agent.git
cd patient-voice-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=your_actual_api_key_here
```

4. Prepare the codebook data:
   - The `data/` folder should contain:
     - `codebook_eppc.json` - EPPC category definitions and examples
     - `codebook_sdoh.json` - SDoH category definitions and examples

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Usage

### Text Input
1. Select the **Text Input** tab
2. Enter or paste patient-provider communication text
3. Click **Analyze**
4. View EPPC and SDoH results

### Voice Input
1. Select the **Voice Input** tab
2. Either:
   - Record live audio using your microphone, OR
   - Upload an audio file (WAV, MP3, OGG, WebM, M4A)
3. Click **Transcribe & Analyze**
4. View transcription and analysis results

### Results History
1. Select the **Results History** tab
2. View:
   - Narrative summary of communication patterns
   - Timeline visualizations
   - Category frequency charts
   - Pattern trends over time

## 🧪 Analyzer Performance

Based on 50 test examples across 10 categories:

| Analyzer | Overall Accuracy | EPPC Accuracy | SDoH Accuracy |
|----------|------------------|---------------|---------------|
| **LLM (GPT-4)** | **82%** | 56% | 100% |
| ML (Embeddings) | 32% | 4% | 68% |
| Trained Model | 22% | 4% | 40% |

**Best Categories:**
- All SDoH categories: 100% with LLM
- Emotional Support: 100%
- Information-Seeking: 80%

## 🏗️ Architecture

```
patient-voice-agent/
├── app.py                      # Main Streamlit application
├── analyzer_llm.py             # LLM-based analyzer (OpenAI)
├── analyzer_ml.py              # ML-based analyzer (embeddings)
├── analyzer_trained.py         # Fine-tuned transformer analyzer
├── voice/
│   ├── __init__.py
│   └── transcription.py        # Audio transcription (Whisper)
├── data/
│   ├── codebook_eppc.json      # EPPC definitions
│   └── codebook_sdoh.json      # SDoH definitions
└── requirements.txt            # Python dependencies
```

## 🔬 Research Alignment

This MVP demonstrates core functionality for:
- **EPPCminer**: Extracting electronic patient-provider communication codes
- **PVminer**: Mining patient voice (SDoH + PC) from text
- Clinical communication quality assessment
- Real-time patient needs identification

**Publications & Proposals:**
- EPPCminer: Patient-provider communication analysis in secure messages
- PVminer: Social determinants of health extraction from patient-generated text

## 🛠️ Technical Details

### Analyzers

**LLM Analyzer:**
- Model: GPT-4o-mini
- Approach: Few-shot learning with codebook examples
- Best for: Semantic understanding and nuanced patterns

**ML Analyzer:**
- Model: all-MiniLM-L6-v2
- Approach: Sentence embeddings + cosine similarity
- Best for: Fast analysis, offline capability

**Trained Analyzer:**
- Model: Fine-tuned Bio_ClinicalBERT
- Training: 1,496 examples from EPPC/SDoH codebooks
- Best for: Domain-specific patterns (Transportation, Employment)

### Voice Processing
- Transcription: OpenAI Whisper API
- Supported formats: WAV, MP3, OGG, WebM, M4A
- Language: English (configurable)

## 📈 Future Enhancements

- [ ] Integration with EHR systems
- [ ] Real-time secure messaging analysis
- [ ] Provider dashboard for population health
- [ ] Multi-language support
- [ ] Batch processing of messages
- [ ] Export reports and analytics

## 🤝 Contributing

This is a research MVP. For collaboration or questions, please contact the research team.

## 📄 License

This project is part of research at Yale University. Please contact the research team for usage permissions.

## 🙏 Acknowledgments

- Yale New Haven Health System
- Cleveland Clinic
- Veterans Administration
- Research team: Dr. Samah Fodeh and collaborators

## 📧 Contact

For questions or collaboration:
- Research Team: [Dr. Samah Fodeh's Lab]
- GitHub: [Your GitHub]

---

**Note:** This is an MVP (Minimum Viable Product) for research demonstration. The trained model has limited accuracy (65%) due to class imbalance in training data. LLM analyzer is recommended for production use.
