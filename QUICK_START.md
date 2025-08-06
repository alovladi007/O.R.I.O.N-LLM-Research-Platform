# ORION Platform - Quick Start Guide

## What is ORION?

ORION is an AI-powered research platform for materials science that helps you:
- 🔍 Search and analyze materials data
- 🧪 Generate new material candidates with AI
- 🔬 Run simulations (DFT, MD, etc.)
- 📋 Generate experimental protocols
- 💬 Chat with an AI assistant about materials science

## How to Run ORION

### Prerequisites
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- OpenAI API key (for AI features)

### Method 1: Quick Local Setup (5 minutes)

1. **Clone the repository**
   ```bash
   git clone https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform.git
   cd O.R.I.O.N-LLM-Research-Platform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file and add your API keys:
   # - OPENAI_API_KEY=your_openai_key_here
   ```

5. **Run the platform**
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

6. **Open in browser**
   - The app will automatically open at http://localhost:8501
   - If not, manually open this URL in your browser

### Method 2: Docker Setup (10 minutes)

1. **Install Docker**
   - Download from: https://www.docker.com/products/docker-desktop

2. **Clone and run**
   ```bash
   git clone https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform.git
   cd O.R.I.O.N-LLM-Research-Platform
   
   # Copy environment file
   cp .env.example .env
   # Edit .env and add your API keys
   
   # Run with Docker
   docker-compose up
   ```

3. **Access the platform**
   - Open http://localhost:8501 in your browser

## What You'll See

When ORION starts, you'll see:

### 1. Chat Interface (Default)
- Talk to the AI about materials science
- Ask questions like:
  - "Find materials with bandgap between 2-3 eV"
  - "How to synthesize TiO2 nanoparticles?"
  - "Generate candidates for solar cell materials"

### 2. Dashboard
- System performance metrics
- Materials analytics
- Research trends

### 3. Knowledge Graph Explorer
- Search materials database
- View properties and relationships

### 4. Candidate Generation
- AI-powered material discovery
- Property prediction with uncertainty

### 5. Simulation Management
- Submit DFT/MD simulations
- Track job status

### 6. Protocol Generation
- Automated experimental procedures
- Safety guidelines

## Troubleshooting

### "Page not loading"
- Make sure you ran `streamlit run src/ui/streamlit_app.py`
- Check if port 8501 is free
- Try: `streamlit run src/ui/streamlit_app.py --server.port 8502`

### "Import errors"
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### "API errors"
- Check your .env file has valid API keys
- OpenAI API key is required for AI features

## Demo Mode

To try without API keys:
```bash
# Run in demo mode (limited features)
streamlit run src/ui/streamlit_app.py -- --demo
```

## Need Help?

1. Check the full documentation: [README.md](README.md)
2. View example usage: [examples/quick_start.py](examples/quick_start.py)
3. Report issues: https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform/issues