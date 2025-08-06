"""
ORION Dashboard
==============

Analytics dashboard for the ORION platform.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard for ORION platform analytics"""
    
    def __init__(self, orion_system):
        self.orion_system = orion_system
        
    def render(self):
        """Render dashboard"""
        st.header("📊 ORION Analytics Dashboard")
        
        # Tabs for different dashboard views
        tabs = st.tabs(["Overview", "Performance", "Materials", "Simulations", "Research Trends"])
        
        with tabs[0]:
            self.render_overview()
            
        with tabs[1]:
            self.render_performance_metrics()
            
        with tabs[2]:
            self.render_materials_analytics()
            
        with tabs[3]:
            self.render_simulation_analytics()
            
        with tabs[4]:
            self.render_research_trends()
    
    def render_overview(self):
        """Render system overview"""
        st.subheader("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Materials",
                "12,456",
                "+234 this week",
                help="Total materials in knowledge graph"
            )
            
        with col2:
            st.metric(
                "Active Simulations",
                "8",
                "+3 today",
                help="Currently running simulations"
            )
            
        with col3:
            st.metric(
                "Candidates Generated",
                "1,234",
                "+56 today",
                help="AI-generated material candidates"
            )
            
        with col4:
            st.metric(
                "Success Rate",
                "78.5%",
                "+2.3%",
                help="Prediction accuracy rate"
            )
        
        # Activity timeline
        st.subheader("Recent Activity")
        
        # Generate sample activity data
        activities = self.generate_activity_data()
        
        fig = go.Figure()
        
        for activity_type in activities['type'].unique():
            data = activities[activities['type'] == activity_type]
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['count'],
                mode='lines+markers',
                name=activity_type,
                stackgroup='one'
            ))
        
        fig.update_layout(
            title="Platform Activity (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="Activity Count",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent discoveries
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Recent Discoveries")
            discoveries = [
                {"material": "Ti0.9Zr0.1O2", "property": "Bandgap", "value": "3.4 eV", "date": "Today"},
                {"material": "MoS2/Graphene", "property": "Conductivity", "value": "1.2×10⁶ S/m", "date": "Yesterday"},
                {"material": "CsPbBr3", "property": "Stability", "value": "92%", "date": "2 days ago"}
            ]
            
            for disc in discoveries:
                with st.container():
                    st.write(f"**{disc['material']}**")
                    st.caption(f"{disc['property']}: {disc['value']} • {disc['date']}")
                    
        with col2:
            st.subheader("🔔 System Alerts")
            alerts = [
                {"type": "info", "message": "Knowledge graph update completed", "time": "10 min ago"},
                {"type": "warning", "message": "High GPU usage detected (92%)", "time": "1 hour ago"},
                {"type": "success", "message": "DFT simulation converged", "time": "2 hours ago"}
            ]
            
            for alert in alerts:
                if alert["type"] == "info":
                    st.info(f"{alert['message']} • {alert['time']}")
                elif alert["type"] == "warning":
                    st.warning(f"{alert['message']} • {alert['time']}")
                elif alert["type"] == "success":
                    st.success(f"{alert['message']} • {alert['time']}")
    
    def render_performance_metrics(self):
        """Render performance analytics"""
        st.subheader("System Performance Metrics")
        
        # Real-time metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Resource utilization over time
            resource_data = self.generate_resource_data()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=resource_data['timestamp'],
                y=resource_data['cpu'],
                mode='lines',
                name='CPU %',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=resource_data['timestamp'],
                y=resource_data['memory'],
                mode='lines',
                name='Memory %',
                line=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=resource_data['timestamp'],
                y=resource_data['gpu'],
                mode='lines',
                name='GPU %',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Resource Utilization (Last Hour)",
                xaxis_title="Time",
                yaxis_title="Usage %",
                yaxis_range=[0, 100],
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Current metrics
            st.subheader("Current Status")
            
            # CPU gauge
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number",
                value=65,
                title={'text': "CPU Usage"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_cpu.update_layout(height=200)
            st.plotly_chart(fig_cpu, use_container_width=True)
            
            # Memory gauge
            fig_mem = go.Figure(go.Indicator(
                mode="gauge+number",
                value=72,
                title={'text': "Memory Usage"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig_mem.update_layout(height=200)
            st.plotly_chart(fig_mem, use_container_width=True)
        
        # Bottleneck analysis
        st.subheader("Bottleneck Analysis")
        
        bottlenecks = [
            {"component": "Knowledge Graph Query", "latency": 245, "calls": 1234},
            {"component": "DFT Calculations", "latency": 8920, "calls": 45},
            {"component": "ML Inference", "latency": 89, "calls": 5678},
            {"component": "RAG Retrieval", "latency": 156, "calls": 2345}
        ]
        
        df_bottlenecks = pd.DataFrame(bottlenecks)
        
        fig = px.scatter(df_bottlenecks, 
                        x='calls', 
                        y='latency',
                        size='latency',
                        color='component',
                        hover_data=['component', 'latency', 'calls'],
                        title="Component Performance Analysis",
                        labels={'calls': 'Number of Calls', 'latency': 'Average Latency (ms)'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_materials_analytics(self):
        """Render materials analytics"""
        st.subheader("Materials Analytics")
        
        # Property distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Bandgap distribution
            bandgap_data = np.random.lognormal(0.5, 0.8, 1000)
            fig = go.Figure(data=[go.Histogram(x=bandgap_data, nbinsx=50)])
            fig.update_layout(
                title="Bandgap Distribution",
                xaxis_title="Bandgap (eV)",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Formation energy vs stability
            n_points = 200
            formation_energy = np.random.normal(-2, 1.5, n_points)
            stability = 1 / (1 + np.exp(formation_energy))
            
            fig = go.Figure(data=go.Scatter(
                x=formation_energy,
                y=stability,
                mode='markers',
                marker=dict(
                    size=8,
                    color=stability,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Stability")
                ),
                text=[f"Material {i}" for i in range(n_points)],
                hovertemplate="Formation Energy: %{x:.2f} eV/atom<br>Stability: %{y:.2f}<br>%{text}"
            ))
            
            fig.update_layout(
                title="Formation Energy vs Stability",
                xaxis_title="Formation Energy (eV/atom)",
                yaxis_title="Stability Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Material families
        st.subheader("Material Families")
        
        families = {
            'Oxides': 4532,
            'Sulfides': 1234,
            'Nitrides': 987,
            'Carbides': 654,
            'Halides': 2345,
            'Intermetallics': 1876,
            '2D Materials': 432,
            'Perovskites': 765
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(families.keys()), y=list(families.values()))
        ])
        
        fig.update_layout(
            title="Materials by Family",
            xaxis_title="Material Family",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions accuracy
        st.subheader("Prediction Accuracy Trends")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        properties = ['Bandgap', 'Formation Energy', 'Bulk Modulus', 'Density']
        
        accuracy_data = pd.DataFrame({
            'Date': dates,
            **{prop: np.random.uniform(0.7, 0.95, 30) + np.random.normal(0, 0.02, 30) 
               for prop in properties}
        })
        
        fig = go.Figure()
        for prop in properties:
            fig.add_trace(go.Scatter(
                x=accuracy_data['Date'],
                y=accuracy_data[prop],
                mode='lines+markers',
                name=prop
            ))
        
        fig.update_layout(
            title="Prediction Accuracy by Property Type",
            xaxis_title="Date",
            yaxis_title="R² Score",
            yaxis_range=[0.6, 1.0],
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_simulation_analytics(self):
        """Render simulation analytics"""
        st.subheader("Simulation Analytics")
        
        # Simulation statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Simulations", "3,456", "+123 this month")
        with col2:
            st.metric("Average Runtime", "4.2 hours", "-0.5 hours")
        with col3:
            st.metric("Convergence Rate", "89.2%", "+1.5%")
        with col4:
            st.metric("Queue Length", "23", "+5")
        
        # Simulation types
        col1, col2 = st.columns(2)
        
        with col1:
            sim_types = {
                'DFT (VASP)': 45,
                'DFT (QE)': 32,
                'MD (LAMMPS)': 18,
                'Monte Carlo': 5
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(sim_types.keys()),
                values=list(sim_types.values()),
                hole=.3
            )])
            
            fig.update_layout(
                title="Simulation Types Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Computational cost
            compute_data = pd.DataFrame({
                'Method': ['DFT-GGA', 'DFT-Hybrid', 'GW', 'MD-Classical', 'MD-Ab initio'],
                'CPU_Hours': [100, 500, 2000, 50, 1000],
                'Accuracy': [0.85, 0.92, 0.96, 0.75, 0.94]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=compute_data['CPU_Hours'],
                y=compute_data['Accuracy'],
                mode='markers+text',
                text=compute_data['Method'],
                textposition="top center",
                marker=dict(
                    size=20,
                    color=compute_data['CPU_Hours'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="CPU Hours")
                )
            ))
            
            fig.update_layout(
                title="Computational Cost vs Accuracy",
                xaxis_title="CPU Hours",
                yaxis_title="Accuracy",
                xaxis_type="log",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Queue status
        st.subheader("Simulation Queue Status")
        
        queue_data = pd.DataFrame({
            'Job_ID': [f'SIM_{i:04d}' for i in range(1, 11)],
            'Material': ['TiO2', 'MoS2', 'GaN', 'Si', 'Graphene', 'BaTiO3', 'ZnO', 'CsPbBr3', 'Fe2O3', 'AlN'],
            'Type': ['DFT', 'MD', 'DFT', 'DFT', 'MD', 'DFT', 'DFT', 'MD', 'DFT', 'DFT'],
            'Status': ['Running', 'Running', 'Queued', 'Queued', 'Running', 'Completed', 'Failed', 'Running', 'Queued', 'Queued'],
            'Progress': [75, 45, 0, 0, 82, 100, 0, 23, 0, 0],
            'Est_Time': ['1.2h', '3.5h', '2.0h', '1.5h', '4.0h', '-', '-', '5.2h', '2.5h', '1.8h']
        })
        
        # Color code by status
        def status_color(status):
            colors = {
                'Running': 'blue',
                'Queued': 'gray',
                'Completed': 'green',
                'Failed': 'red'
            }
            return colors.get(status, 'gray')
        
        st.dataframe(
            queue_data.style.applymap(
                lambda x: f"color: {status_color(x)}" if x in ['Running', 'Queued', 'Completed', 'Failed'] else "",
                subset=['Status']
            ),
            use_container_width=True
        )
    
    def render_research_trends(self):
        """Render research trends and insights"""
        st.subheader("Research Trends & Insights")
        
        # Hot topics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Trending materials/properties
            topics = ['2D Materials', 'Perovskites', 'High-k Dielectrics', 'Photocatalysts', 
                     'Thermoelectrics', 'Quantum Materials', 'Topological Insulators', 'MOFs']
            mentions = [234, 189, 156, 143, 128, 112, 98, 87]
            growth = [15, 23, -5, 12, 8, 45, 32, 18]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=topics,
                y=mentions,
                name='Mentions',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Scatter(
                x=topics,
                y=growth,
                name='Growth %',
                yaxis='y2',
                mode='lines+markers',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Trending Research Topics (Last 30 Days)",
                xaxis_title="Topic",
                yaxis_title="Number of Mentions",
                yaxis2=dict(
                    title="Growth Rate %",
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Top Properties")
            properties = {
                'Bandgap': 456,
                'Conductivity': 234,
                'Stability': 189,
                'Efficiency': 167,
                'Hardness': 134
            }
            
            for prop, count in properties.items():
                st.metric(prop, count, f"+{np.random.randint(5, 20)}")
        
        # Collaboration network
        st.subheader("Research Collaboration Network")
        
        # Sample network data
        nodes = ['MIT', 'Stanford', 'Berkeley', 'NREL', 'Argonne', 'ORNL', 'Caltech', 'Northwestern']
        edges = [
            ('MIT', 'Stanford', 15),
            ('MIT', 'NREL', 8),
            ('Stanford', 'Berkeley', 12),
            ('Berkeley', 'NREL', 6),
            ('NREL', 'Argonne', 10),
            ('Argonne', 'ORNL', 7),
            ('Caltech', 'Stanford', 9),
            ('Northwestern', 'Argonne', 11)
        ]
        
        # Create network visualization
        edge_trace = []
        for edge in edges:
            x0, y0 = np.random.rand(2)
            x1, y1 = np.random.rand(2)
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge[2]/3, color='gray'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=np.random.rand(len(nodes)),
            y=np.random.rand(len(nodes)),
            mode='markers+text',
            text=nodes,
            textposition="top center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hovertext=[f"{node}: {np.random.randint(20, 100)} publications" for node in nodes],
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title="Institution Collaboration Network",
            showlegend=False,
            hovermode='closest',
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_activity_data(self):
        """Generate sample activity data"""
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        
        data = []
        for date in dates:
            for activity_type in ['Searches', 'Generations', 'Simulations']:
                count = np.random.poisson(20 if activity_type == 'Searches' else 10)
                data.append({
                    'timestamp': date,
                    'type': activity_type,
                    'count': count
                })
        
        return pd.DataFrame(data)
    
    def generate_resource_data(self):
        """Generate sample resource utilization data"""
        timestamps = pd.date_range(end=datetime.now(), periods=60, freq='min')
        
        data = {
            'timestamp': timestamps,
            'cpu': np.random.normal(65, 10, 60).clip(0, 100),
            'memory': np.random.normal(72, 8, 60).clip(0, 100),
            'gpu': np.random.normal(45, 15, 60).clip(0, 100)
        }
        
        return pd.DataFrame(data)