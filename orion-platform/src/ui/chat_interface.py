"""
ORION Chat Interface
===================

Natural language interface for interacting with ORION.
"""

import streamlit as st
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class ChatInterface:
    """Chat interface for ORION platform"""
    
    def __init__(self, orion_system):
        self.orion_system = orion_system
        self.suggested_queries = [
            "Find materials with bandgap between 2-3 eV",
            "How to synthesize TiO2 nanoparticles?",
            "Generate candidates for high-k dielectrics",
            "Compare properties of anatase vs rutile TiO2",
            "What are applications of graphene in electronics?",
            "Design experiment for perovskite solar cells",
            "Predict stability of Cs2AgBiBr6",
            "Search recent papers on 2D materials"
        ]
        
    def render(self):
        """Render chat interface"""
        st.header("💬 ORION Chat Assistant")
        
        # Display suggested queries for new users
        if not st.session_state.messages:
            st.info("Welcome to ORION! I can help you with materials discovery, property prediction, "
                   "synthesis planning, and more. Try one of these queries or ask your own question.")
            
            cols = st.columns(2)
            for i, query in enumerate(self.suggested_queries):
                col = cols[i % 2]
                with col:
                    if st.button(query, key=f"suggest_{i}", use_container_width=True):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": query,
                            "timestamp": datetime.now()
                        })
                        asyncio.run(self.process_message(query))
                        st.rerun()
        
        # Chat history
        self.display_chat_history()
        
        # Chat input
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.chat_input("Ask ORION anything about materials science...")
                
            with col2:
                if st.button("🗑️ Clear", help="Clear chat history"):
                    st.session_state.messages = []
                    st.rerun()
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Process message
            asyncio.run(self.process_message(user_input))
            st.rerun()
    
    def display_chat_history(self):
        """Display chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    # Assistant message with rich formatting
                    self.render_assistant_message(message)
                
                # Timestamp
                st.caption(f"{message['timestamp'].strftime('%H:%M:%S')}")
    
    def render_assistant_message(self, message: Dict[str, Any]):
        """Render assistant message with rich content"""
        # Main response text
        st.write(message["content"])
        
        # Additional data visualization
        if "data" in message:
            data = message["data"]
            
            # Materials found
            if "materials" in data:
                with st.expander(f"📊 Found {len(data['materials'])} materials"):
                    for material in data['materials']:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{material['name']}** ({material['formula']})")
                            if 'properties' in material:
                                for prop, value in material['properties'].items():
                                    st.write(f"• {prop}: {value}")
                        with col2:
                            if st.button("Details", key=f"detail_{material['id']}"):
                                st.session_state.selected_material = material
            
            # Candidates generated
            if "candidates" in data:
                with st.expander(f"🧪 Generated {len(data['candidates'])} candidates"):
                    df_data = []
                    for cand in data['candidates']:
                        df_data.append({
                            'Formula': cand.get('formula', 'Unknown'),
                            'Stability': f"{cand.get('stability_score', 0):.2f}",
                            'Target Property': f"{cand.get('target_value', 0):.2f}",
                            'Uncertainty': f"±{cand.get('uncertainty', 0):.2f}",
                            'Physics Valid': '✅' if cand.get('physics_valid', False) else '❌'
                        })
                    
                    if df_data:
                        st.dataframe(df_data, use_container_width=True)
            
            # Simulation results
            if "simulation" in data:
                sim = data["simulation"]
                with st.expander("🔬 Simulation Results"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Energy", f"{sim.get('energy', 0):.3f} eV")
                    with col2:
                        st.metric("Forces", f"{sim.get('max_force', 0):.3f} eV/Å")
                    with col3:
                        st.metric("Convergence", sim.get('converged', 'Unknown'))
            
            # Charts
            if "chart" in data:
                chart_data = data["chart"]
                if chart_data["type"] == "scatter":
                    st.scatter_chart(chart_data["data"])
                elif chart_data["type"] == "line":
                    st.line_chart(chart_data["data"])
                elif chart_data["type"] == "bar":
                    st.bar_chart(chart_data["data"])
            
            # References
            if "references" in data:
                with st.expander("📚 References"):
                    for i, ref in enumerate(data["references"], 1):
                        st.write(f"{i}. {ref.get('title', 'Unknown')}")
                        st.caption(f"   {ref.get('authors', 'Unknown')} - {ref.get('year', 'Unknown')}")
                        if 'doi' in ref:
                            st.caption(f"   DOI: {ref['doi']}")
        
        # Action buttons
        if "actions" in message:
            cols = st.columns(len(message["actions"]))
            for i, action in enumerate(message["actions"]):
                with cols[i]:
                    if st.button(action["label"], key=f"action_{message['timestamp']}_{i}"):
                        if action["type"] == "generate":
                            st.session_state.current_view = 'candidates'
                        elif action["type"] == "simulate":
                            st.session_state.current_view = 'simulations'
                        elif action["type"] == "protocol":
                            st.session_state.current_view = 'protocols'
                        st.rerun()
    
    async def process_message(self, user_input: str):
        """Process user message and generate response"""
        with st.spinner("ORION is thinking..."):
            try:
                # Process query through ORION system
                response = await self.orion_system.process_query(user_input)
                
                # Create assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("text", "I'm processing your request..."),
                    "timestamp": datetime.now()
                }
                
                # Add data if available
                if "data" in response:
                    assistant_message["data"] = response["data"]
                
                # Add suggested actions
                assistant_message["actions"] = self.get_suggested_actions(response)
                
                # Add to messages
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_message = {
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                    "timestamp": datetime.now()
                }
                st.session_state.messages.append(error_message)
    
    def get_suggested_actions(self, response: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get suggested actions based on response type"""
        actions = []
        
        response_type = response.get("type", "general")
        
        if response_type == "material_search":
            actions.append({
                "label": "Generate Similar Materials",
                "type": "generate"
            })
            actions.append({
                "label": "Run Simulations",
                "type": "simulate"
            })
            
        elif response_type == "candidate_generation":
            actions.append({
                "label": "Simulate Top Candidates",
                "type": "simulate"
            })
            actions.append({
                "label": "Generate Protocols",
                "type": "protocol"
            })
            
        elif response_type == "synthesis":
            actions.append({
                "label": "Generate Full Protocol",
                "type": "protocol"
            })
            actions.append({
                "label": "Find Similar Methods",
                "type": "search"
            })
        
        return actions
    
    def render_typing_indicator(self):
        """Render typing indicator animation"""
        typing_placeholder = st.empty()
        with typing_placeholder.container():
            st.write("ORION is typing...")
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    st.write("●" if i == st.session_state.get('typing_dot', 0) else "○")
        
        # Update animation state
        if 'typing_dot' not in st.session_state:
            st.session_state.typing_dot = 0
        st.session_state.typing_dot = (st.session_state.typing_dot + 1) % 3