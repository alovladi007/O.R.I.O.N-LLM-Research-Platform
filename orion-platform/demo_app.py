"""
ORION Demo - Simplified Version
==============================

This is a demo version that works with minimal dependencies.
Run with: streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ORION Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Chat'

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=ORION", width=300)
    st.title("ORION Platform")
    st.markdown("*Materials Science AI Assistant*")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Chat", "Dashboard", "Material Search", "Generate Candidates"]
    )
    st.session_state.current_page = page
    
    st.divider()
    
    # Info
    st.info("This is a demo version with limited features. Full version requires additional setup.")

# Main content
if st.session_state.current_page == "Chat":
    st.header("💬 ORION Chat Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about materials science..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate demo response
        response = f"I understand you're asking about: '{prompt}'. In the full version, I would provide detailed materials science insights using AI. This demo shows the interface structure."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

elif st.session_state.current_page == "Dashboard":
    st.header("📊 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Materials in Database", "12,456", "+234")
    with col2:
        st.metric("Active Simulations", "8", "+3")
    with col3:
        st.metric("Candidates Generated", "1,234", "+56")
    with col4:
        st.metric("Success Rate", "78.5%", "+2.3%")
    
    # Sample chart
    st.subheader("Material Properties Distribution")
    
    # Generate sample data
    materials = pd.DataFrame({
        'Material': [f'Material_{i}' for i in range(50)],
        'Bandgap': np.random.lognormal(0.5, 0.8, 50),
        'Formation_Energy': np.random.normal(-2, 1, 50),
        'Stability': np.random.uniform(0, 1, 50)
    })
    
    fig = px.scatter(materials, x='Formation_Energy', y='Bandgap', 
                     color='Stability', size='Stability',
                     hover_data=['Material'],
                     title="Materials Property Space")
    
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == "Material Search":
    st.header("🔍 Material Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search materials...", placeholder="e.g., TiO2, perovskite, bandgap > 2")
    
    with col2:
        if st.button("Search", type="primary"):
            st.success("Search would be performed in full version")
    
    # Demo results
    st.subheader("Sample Results")
    
    demo_results = pd.DataFrame({
        'Material': ['TiO2', 'ZnO', 'GaN', 'Si', 'GaAs'],
        'Formula': ['TiO2', 'ZnO', 'GaN', 'Si', 'GaAs'],
        'Bandgap (eV)': [3.2, 3.4, 3.4, 1.1, 1.4],
        'Crystal System': ['Tetragonal', 'Hexagonal', 'Hexagonal', 'Cubic', 'Cubic'],
        'Stability': ['Stable', 'Stable', 'Stable', 'Stable', 'Stable']
    })
    
    st.dataframe(demo_results, use_container_width=True)

elif st.session_state.current_page == "Generate Candidates":
    st.header("⚗️ Generate Material Candidates")
    
    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            target_property = st.selectbox("Target Property", ["Bandgap", "Conductivity", "Hardness"])
            target_value = st.number_input("Target Value", min_value=0.0, max_value=10.0, value=2.5)
        
        with col2:
            num_candidates = st.slider("Number of Candidates", 1, 20, 5)
            use_ml = st.checkbox("Use ML Predictions", value=True)
        
        submitted = st.form_submit_button("Generate Candidates", type="primary")
        
        if submitted:
            st.success("In the full version, AI would generate novel material candidates here!")
            
            # Demo candidates
            st.subheader("Generated Candidates (Demo)")
            
            candidates = pd.DataFrame({
                'Candidate': [f'Candidate_{i+1}' for i in range(num_candidates)],
                'Formula': ['Ti0.9Zr0.1O2', 'Zn0.8Mg0.2O', 'Ga0.7Al0.3N', 'Si0.5Ge0.5', 'In0.6Ga0.4As'][:num_candidates],
                f'{target_property} (predicted)': np.random.normal(target_value, 0.3, num_candidates),
                'Stability Score': np.random.uniform(0.7, 0.95, num_candidates),
                'Confidence': np.random.uniform(0.6, 0.9, num_candidates)
            })
            
            st.dataframe(candidates, use_container_width=True)

# Footer
st.divider()
st.caption("ORION Demo Version - Full features available in complete installation")
st.caption("Repository: https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform")