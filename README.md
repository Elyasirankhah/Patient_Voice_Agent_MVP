# 🎤 Patient Voice Agent MVP

AI-powered analysis of patient-provider communication and social determinants of health from text and voice input.

## Features

- 🎙️ **Voice Input**: Live recording and transcription (OpenAI Whisper)
- 💬 **Text Analysis**: Direct text input analysis
- 🤖 **Multiple AI Models**: LLM (GPT-4), ML embeddings, fine-tuned Bio_ClinicalBERT
- 📊 **Visual Analytics**: Narrative summaries, timelines, and pattern trends
- 🏥 **Dual Analysis**:
  - **EPPC**: Partnership, Emotional Support, Information-Giving, Information-Seeking, Shared Decision-Making
  - **SDoH**: Housing, Financial, Food, Transportation, Employment

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up API key:**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

3. **Run:**
```bash
streamlit run app.py
```

> **Note:** Codebook data is not included in this repo. For deployment, add codebooks and API key to Streamlit secrets.

## Deployment

For Streamlit Cloud deployment:
- Add `OPENAI_API_KEY` to secrets
- Add codebooks to secrets (app auto-loads from secrets if local files missing)

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **ML**: Sentence Transformers (MiniLM, Bio_ClinicalBERT)
- **Voice**: OpenAI Whisper API

## Research Context

Supports clinical communication research:
- EPPCminer: Patient-provider communication analysis
- PVminer: Social determinants of health extraction

---

**Yale University Research Project**
