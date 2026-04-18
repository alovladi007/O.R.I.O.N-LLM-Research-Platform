# macOS Setup Guide for ORION

## Fix for your current error

The error you're seeing is because some dependencies need to be compiled. Here's how to fix it:

### Option 1: Quick Fix (Recommended)

1. **Install Homebrew** (if you don't have it):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install cmake**:
   ```bash
   brew install cmake
   ```

3. **Use minimal requirements**:
   ```bash
   cd O.R.I.O.N-LLM-Research-Platform
   pip install -r requirements-minimal.txt
   ```

4. **Run ORION**:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

### Option 2: Using Conda (Your current setup)

Since you're using Anaconda, try this:

```bash
# Activate your conda environment
conda activate base

# Install dependencies via conda
conda install -c conda-forge streamlit plotly pandas numpy python-dotenv pyyaml

# Navigate to ORION directory
cd O.R.I.O.N-LLM-Research-Platform

# Run ORION
streamlit run src/ui/streamlit_app.py
```

### Option 3: Create a new conda environment

```bash
# Create fresh environment
conda create -n orion python=3.10
conda activate orion

# Install minimal requirements
conda install -c conda-forge streamlit plotly pandas numpy

# Run ORION
cd O.R.I.O.N-LLM-Research-Platform
streamlit run src/ui/streamlit_app.py
```

## If streamlit still not found

After installation, try:
```bash
# Find where streamlit was installed
which streamlit

# Or with conda
conda list | grep streamlit

# Run with full path if needed
python -m streamlit run src/ui/streamlit_app.py
```

## Simplest Test

To just see if it works, create a test file:

```bash
# Create a simple test
echo 'import streamlit as st
st.title("ORION is Working!")
st.write("Setup successful! 🎉")' > test_app.py

# Run it
python -m streamlit run test_app.py
```

## Still having issues?

Try the web-based option:
1. Use GitHub Codespaces (free tier available)
2. Or use Google Colab
3. Or use Streamlit Cloud (deploy directly from GitHub)

The full ORION platform has many dependencies that can be complex to install. The minimal version will let you see the UI and basic features!