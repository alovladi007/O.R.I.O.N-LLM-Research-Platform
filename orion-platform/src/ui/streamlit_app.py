"""
ORION Streamlit Web Application
==============================

Main entry point for the ORION web interface.
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ui.chat_interface import ChatInterface
from src.ui.dashboard import Dashboard
from src.core import ORIONSystem, ConfigManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamlitApp:
    """Main Streamlit application for ORION"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.orion_system = None
        self.chat_interface = None
        self.dashboard = None
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.messages = []
            st.session_state.current_view = 'chat'
            st.session_state.system_status = {}
            st.session_state.search_results = []
            st.session_state.candidates = []
            st.session_state.simulations = []
            
    async def initialize_system(self):
        """Initialize ORION system"""
        if not st.session_state.initialized:
            with st.spinner("Initializing ORION system..."):
                self.orion_system = ORIONSystem(self.config)
                await self.orion_system.initialize()
                
                self.chat_interface = ChatInterface(self.orion_system)
                self.dashboard = Dashboard(self.orion_system)
                
                st.session_state.initialized = True
                st.session_state.system_status = await self.orion_system.get_system_status()
                
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.image("https://via.placeholder.com/300x100?text=ORION", width=300)
            st.title("ORION Platform")
            st.markdown("*Optimized Research & Innovation for Organized Nanomaterials*")
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            if st.button("💬 Chat Interface", use_container_width=True):
                st.session_state.current_view = 'chat'
                
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.current_view = 'dashboard'
                
            if st.button("🔍 Knowledge Graph", use_container_width=True):
                st.session_state.current_view = 'knowledge_graph'
                
            if st.button("⚗️ Candidate Generation", use_container_width=True):
                st.session_state.current_view = 'candidates'
                
            if st.button("🧪 Simulations", use_container_width=True):
                st.session_state.current_view = 'simulations'
                
            if st.button("📋 Protocols", use_container_width=True):
                st.session_state.current_view = 'protocols'
                
            st.divider()
            
            # System Status
            st.subheader("System Status")
            if st.session_state.initialized:
                status = st.session_state.system_status
                
                # Performance metrics
                if 'performance' in status:
                    perf = status['performance']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CPU", f"{perf.get('cpu_usage', 0):.1f}%")
                        st.metric("Memory", f"{perf.get('memory_usage', 0):.1f}%")
                    with col2:
                        st.metric("GPU", f"{perf.get('gpu_usage', 0):.1f}%")
                        st.metric("Queue", perf.get('queue_size', 0))
                
                # Knowledge Graph stats
                if 'knowledge_graph' in status:
                    kg = status['knowledge_graph']
                    st.metric("Materials", kg.get('total_materials', 0))
                    st.metric("Properties", kg.get('total_properties', 0))
            else:
                st.info("System initializing...")
                
            st.divider()
            
            # Settings
            with st.expander("⚙️ Settings"):
                st.slider("Temperature", 0.0, 2.0, 0.7, key="llm_temperature")
                st.slider("Max Results", 5, 50, 10, key="max_results")
                st.checkbox("Enable Physics Validation", value=True, key="physics_validation")
                st.checkbox("Show Uncertainty", value=True, key="show_uncertainty")
                
    def render_main_content(self):
        """Render main content based on current view"""
        if st.session_state.current_view == 'chat':
            self.chat_interface.render()
            
        elif st.session_state.current_view == 'dashboard':
            self.dashboard.render()
            
        elif st.session_state.current_view == 'knowledge_graph':
            self.render_knowledge_graph_view()
            
        elif st.session_state.current_view == 'candidates':
            self.render_candidates_view()
            
        elif st.session_state.current_view == 'simulations':
            self.render_simulations_view()
            
        elif st.session_state.current_view == 'protocols':
            self.render_protocols_view()
            
    def render_knowledge_graph_view(self):
        """Render knowledge graph exploration view"""
        st.header("🔍 Knowledge Graph Explorer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search interface
            search_query = st.text_input("Search materials, properties, or methods:", 
                                       placeholder="e.g., TiO2 bandgap, perovskite synthesis")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                search_type = st.selectbox("Search Type", 
                                         ["All", "Materials", "Properties", "Methods", "Publications"])
            with col_b:
                property_filter = st.selectbox("Property Filter",
                                             ["Any", "Bandgap", "Formation Energy", "Bulk Modulus"])
            with col_c:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5)
            
            if st.button("Search", type="primary", use_container_width=True):
                asyncio.run(self.search_knowledge_graph(search_query, search_type, 
                                                       property_filter, min_confidence))
        
        with col2:
            # Statistics
            st.subheader("Graph Statistics")
            if st.session_state.initialized:
                stats = st.session_state.system_status.get('knowledge_graph', {})
                st.metric("Total Nodes", stats.get('total_nodes', 0))
                st.metric("Total Edges", stats.get('total_edges', 0))
                st.metric("Data Sources", stats.get('num_sources', 0))
        
        # Results
        if st.session_state.search_results:
            st.subheader("Search Results")
            for result in st.session_state.search_results:
                with st.expander(f"{result['name']} - {result['type']}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Formula:** {result.get('formula', 'N/A')}")
                        st.write(f"**Description:** {result.get('description', 'N/A')}")
                        
                        if 'properties' in result:
                            st.write("**Properties:**")
                            for prop, value in result['properties'].items():
                                st.write(f"- {prop}: {value}")
                    
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                        if st.button(f"Generate Candidates", key=f"gen_{result['id']}"):
                            st.session_state.current_view = 'candidates'
                            st.rerun()
                            
    def render_candidates_view(self):
        """Render candidate generation view"""
        st.header("⚗️ Material Candidate Generation")
        
        with st.form("candidate_generation"):
            st.subheader("Generation Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                target_property = st.selectbox("Target Property",
                                             ["Bandgap", "Formation Energy", "Bulk Modulus", 
                                              "Thermal Conductivity", "Density"])
                target_value = st.number_input(f"Target {target_property} Value", 
                                             min_value=0.0, max_value=100.0, value=2.0)
                maximize = st.checkbox("Maximize property")
                
            with col2:
                num_candidates = st.slider("Number of Candidates", 1, 20, 5)
                diversity_weight = st.slider("Diversity Weight", 0.0, 1.0, 0.3)
                enable_physics = st.checkbox("Physics Validation", value=True)
            
            constraints = st.text_area("Constraints (JSON format)", 
                                     value='{"forbidden_elements": ["Pb", "Hg", "Cd"]}')
            
            submitted = st.form_submit_button("Generate Candidates", type="primary", 
                                            use_container_width=True)
            
            if submitted:
                asyncio.run(self.generate_candidates(target_property, target_value, 
                                                   maximize, num_candidates, 
                                                   diversity_weight, enable_physics, 
                                                   constraints))
        
        # Display candidates
        if st.session_state.candidates:
            st.subheader("Generated Candidates")
            
            for i, candidate in enumerate(st.session_state.candidates):
                with st.expander(f"Candidate {i+1}: {candidate.get('formula', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Composition:**")
                        for elem, frac in candidate.get('composition', {}).items():
                            st.write(f"- {elem}: {frac:.3f}")
                    
                    with col2:
                        st.write("**Predicted Properties:**")
                        predictions = candidate.get('predictions', {})
                        uncertainties = candidate.get('uncertainties', {})
                        
                        for prop, value in predictions.items():
                            unc = uncertainties.get(prop, 0)
                            st.write(f"- {prop}: {value:.3f} ± {unc:.3f}")
                    
                    with col3:
                        st.metric("Stability Score", f"{candidate.get('stability_score', 0):.2f}")
                        st.metric("Physics Valid", "✅" if candidate.get('physics_valid', False) else "❌")
                        
                        if st.button(f"Simulate", key=f"sim_{candidate['id']}"):
                            st.session_state.current_view = 'simulations'
                            st.rerun()
                            
    def render_simulations_view(self):
        """Render simulations view"""
        st.header("🧪 Simulation Management")
        
        tabs = st.tabs(["Submit New", "Active Jobs", "Completed", "Analysis"])
        
        with tabs[0]:
            st.subheader("Submit New Simulation")
            
            col1, col2 = st.columns(2)
            with col1:
                sim_type = st.selectbox("Simulation Type", 
                                      ["DFT (VASP)", "DFT (Quantum Espresso)", 
                                       "Molecular Dynamics", "Monte Carlo"])
                material_id = st.text_input("Material ID or Formula")
                
            with col2:
                compute_resources = st.selectbox("Compute Resources",
                                               ["Local", "HPC Cluster", "Cloud (AWS)", "Cloud (GCP)"])
                priority = st.select_slider("Priority", ["Low", "Normal", "High", "Urgent"])
            
            parameters = st.text_area("Simulation Parameters (JSON)", 
                                    value='{"kpoints": [4,4,4], "encut": 500}')
            
            if st.button("Submit Simulation", type="primary", use_container_width=True):
                st.success("Simulation submitted successfully!")
                
        with tabs[1]:
            st.subheader("Active Simulation Jobs")
            # Would show active jobs from the system
            st.info("No active simulations")
            
        with tabs[2]:
            st.subheader("Completed Simulations")
            # Would show completed simulations
            st.info("No completed simulations")
            
        with tabs[3]:
            st.subheader("Simulation Analysis")
            # Would show analysis tools
            st.info("Select a completed simulation to analyze")
            
    def render_protocols_view(self):
        """Render experimental protocols view"""
        st.header("📋 Experimental Protocol Generation")
        
        with st.form("protocol_generation"):
            st.subheader("Protocol Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                experiment_type = st.selectbox("Experiment Type",
                                             ["Synthesis", "Characterization", "Device Fabrication"])
                material = st.text_input("Target Material", "TiO2")
                method = st.selectbox("Method", 
                                    ["Sol-gel", "CVD", "ALD", "Hydrothermal", "Solid State"])
                
            with col2:
                scale = st.selectbox("Scale", ["Lab Scale", "Pilot Scale", "Industrial Scale"])
                safety_level = st.selectbox("Safety Requirements", 
                                          ["Standard", "Enhanced", "Clean Room"])
                output_format = st.selectbox("Output Format", ["PDF", "Markdown", "LaTeX"])
            
            additional_notes = st.text_area("Additional Requirements")
            
            submitted = st.form_submit_button("Generate Protocol", type="primary",
                                            use_container_width=True)
            
            if submitted:
                st.success("Protocol generated successfully!")
                
                # Show generated protocol
                st.subheader("Generated Protocol")
                st.markdown("""
                # Synthesis of TiO2 via Sol-gel Method
                
                ## Objective
                To synthesize high-purity TiO2 nanoparticles using sol-gel method.
                
                ## Materials & Reagents
                - Titanium isopropoxide (Ti(OiPr)4): 10 mL
                - Isopropanol: 40 mL
                - Deionized water: 5 mL
                - Nitric acid (0.1 M): 2 mL
                
                ## Equipment
                - Round-bottom flask (100 mL)
                - Magnetic stirrer with hot plate
                - Reflux condenser
                - Furnace (up to 500°C)
                
                ## Procedure
                1. In a fume hood, add 40 mL of isopropanol to the round-bottom flask
                2. Slowly add 10 mL of titanium isopropoxide while stirring
                3. Prepare hydrolysis solution: mix 5 mL water with 2 mL 0.1M HNO3
                4. Add hydrolysis solution dropwise over 30 minutes while stirring vigorously
                5. Continue stirring for 2 hours at room temperature
                6. Heat to 80°C and maintain for 1 hour under reflux
                7. Cool to room temperature and age for 24 hours
                8. Dry at 100°C for 12 hours
                9. Calcine at 450°C for 2 hours (heating rate: 5°C/min)
                
                ## Safety Notes
                - Work in well-ventilated fume hood
                - Wear appropriate PPE (lab coat, gloves, safety glasses)
                - Titanium isopropoxide is moisture sensitive and flammable
                
                ## Expected Results
                - White powder of anatase TiO2
                - Particle size: 10-20 nm
                - Surface area: 50-100 m²/g
                """)
                
                if st.button("Download Protocol"):
                    st.info("Download functionality would be implemented here")
    
    async def search_knowledge_graph(self, query, search_type, property_filter, min_confidence):
        """Search knowledge graph"""
        # This would call the actual ORION system
        # For demo, using mock data
        st.session_state.search_results = [
            {
                'id': 'mat_001',
                'name': 'TiO2 (Anatase)',
                'type': 'Material',
                'formula': 'TiO2',
                'description': 'Titanium dioxide in anatase crystal structure',
                'properties': {
                    'bandgap': '3.2 eV',
                    'density': '3.9 g/cm³',
                    'crystal_system': 'Tetragonal'
                },
                'confidence': 0.95
            },
            {
                'id': 'mat_002',
                'name': 'TiO2 (Rutile)',
                'type': 'Material',
                'formula': 'TiO2',
                'description': 'Titanium dioxide in rutile crystal structure',
                'properties': {
                    'bandgap': '3.0 eV',
                    'density': '4.2 g/cm³',
                    'crystal_system': 'Tetragonal'
                },
                'confidence': 0.92
            }
        ]
        
    async def generate_candidates(self, target_property, target_value, maximize, 
                                num_candidates, diversity_weight, enable_physics, constraints):
        """Generate material candidates"""
        # This would call the actual ORION system
        # For demo, using mock data
        st.session_state.candidates = [
            {
                'id': 'cand_001',
                'formula': 'Ti0.9Zr0.1O2',
                'composition': {'Ti': 0.9, 'Zr': 0.1, 'O': 2.0},
                'predictions': {
                    'bandgap': 3.3,
                    'formation_energy': -9.2,
                    'bulk_modulus': 210
                },
                'uncertainties': {
                    'bandgap': 0.1,
                    'formation_energy': 0.3,
                    'bulk_modulus': 15
                },
                'stability_score': 0.85,
                'physics_valid': True
            },
            {
                'id': 'cand_002',
                'formula': 'Ti0.8Hf0.2O2',
                'composition': {'Ti': 0.8, 'Hf': 0.2, 'O': 2.0},
                'predictions': {
                    'bandgap': 3.4,
                    'formation_energy': -9.0,
                    'bulk_modulus': 220
                },
                'uncertainties': {
                    'bandgap': 0.15,
                    'formation_energy': 0.4,
                    'bulk_modulus': 18
                },
                'stability_score': 0.82,
                'physics_valid': True
            }
        ]
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="ORION Platform",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        self.initialize_session_state()
        
        # Initialize system
        asyncio.run(self.initialize_system())
        
        # Render UI
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main entry point"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()