#!/usr/bin/env python3
"""
ORION Platform Demo Server
==========================

A simple demo server to showcase the ORION platform interface.
"""

import http.server
import socketserver
import os
from pathlib import Path

# HTML content for the demo
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ORION Platform - AI-Driven Materials Science</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: white;
        }
        .header {
            background: rgba(0,0,0,0.2);
            padding: 1rem 2rem;
            backdrop-filter: blur(10px);
        }
        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .nav-links {
            display: flex;
            gap: 2rem;
            list-style: none;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            transition: opacity 0.3s;
        }
        .nav-links a:hover { opacity: 0.8; }
        .hero {
            max-width: 1200px;
            margin: 0 auto;
            padding: 4rem 2rem;
            text-align: center;
        }
        h1 {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 3rem;
        }
        .buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 1rem 2rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.3s, box-shadow 0.3s;
            display: inline-block;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .btn-primary {
            background: white;
            color: #2a5298;
        }
        .btn-secondary {
            background: transparent;
            color: white;
            border: 2px solid white;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1200px;
            margin: 4rem auto;
            padding: 0 2rem;
        }
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 2rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.15);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .feature-title {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            max-width: 800px;
            margin: 4rem auto;
            padding: 2rem;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        .stat {
            text-align: center;
        }
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4fc3f7;
        }
        .stat-label {
            opacity: 0.8;
            margin-top: 0.5rem;
        }
        .demo-section {
            max-width: 1200px;
            margin: 4rem auto;
            padding: 0 2rem;
            text-align: center;
        }
        .demo-terminal {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 2rem 0;
            text-align: left;
            font-family: 'Courier New', monospace;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        .terminal-header {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .terminal-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot-red { background: #ff5f56; }
        .dot-yellow { background: #ffbd2e; }
        .dot-green { background: #27c93f; }
        .terminal-content {
            color: #0f0;
            line-height: 1.6;
        }
        .footer {
            background: rgba(0,0,0,0.3);
            padding: 2rem;
            text-align: center;
            margin-top: 4rem;
        }
    </style>
</head>
<body>
    <header class="header">
        <nav class="nav">
            <div class="logo">
                🔬 ORION Platform
            </div>
            <ul class="nav-links">
                <li><a href="#features">Features</a></li>
                <li><a href="#demo">Demo</a></li>
                <li><a href="#docs">Documentation</a></li>
                <li><a href="#api">API</a></li>
            </ul>
        </nav>
    </header>

    <section class="hero">
        <h1>Accelerate Materials Discovery with AI</h1>
        <p class="subtitle">From theoretical concepts to laboratory protocols in minutes, not months.</p>
        
        <div class="buttons">
            <a href="#demo" class="btn btn-primary">Start Free Trial</a>
            <a href="#features" class="btn btn-secondary">Watch Demo</a>
        </div>
    </section>

    <div class="stats">
        <div class="stat">
            <div class="stat-number">10M+</div>
            <div class="stat-label">Materials in Database</div>
        </div>
        <div class="stat">
            <div class="stat-number">50K+</div>
            <div class="stat-label">Simulations Run</div>
        </div>
        <div class="stat">
            <div class="stat-number">99.9%</div>
            <div class="stat-label">Uptime SLA</div>
        </div>
    </div>

    <section id="features" class="features">
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI-Powered Discovery</h3>
            <p>Leverage cutting-edge LLMs to generate novel material candidates based on desired properties.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3 class="feature-title">Simulation Integration</h3>
            <p>Seamlessly run DFT, molecular dynamics, and FEA simulations with automated workflows.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Knowledge Graph</h3>
            <p>Navigate a comprehensive materials ontology with intelligent relationship mapping.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3 class="feature-title">Real-time Collaboration</h3>
            <p>Work together with your team in real-time with live updates and shared workspaces.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <h3 class="feature-title">Enterprise Security</h3>
            <p>Bank-level encryption, OAuth2 authentication, and fine-grained access control.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">☁️</div>
            <h3 class="feature-title">Cloud-Native</h3>
            <p>Scalable infrastructure that grows with your research needs.</p>
        </div>
    </section>

    <section id="demo" class="demo-section">
        <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">See It In Action</h2>
        <p style="opacity: 0.9; margin-bottom: 2rem;">Generate novel materials with a simple API call</p>
        
        <div class="demo-terminal">
            <div class="terminal-header">
                <span class="terminal-dot dot-red"></span>
                <span class="terminal-dot dot-yellow"></span>
                <span class="terminal-dot dot-green"></span>
            </div>
            <div class="terminal-content">
$ curl -X POST https://api.orion-platform.ai/v1/materials/generate \\<br>
&nbsp;&nbsp;-H "Authorization: Bearer YOUR_API_KEY" \\<br>
&nbsp;&nbsp;-d '{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"description": "High-temperature superconductor with Tc > 100K",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"constraints": {"elements": ["Cu", "O", "Y", "Ba"]},<br>
&nbsp;&nbsp;&nbsp;&nbsp;"num_candidates": 5<br>
&nbsp;&nbsp;}'<br><br>
{<br>
&nbsp;&nbsp;"candidates": [<br>
&nbsp;&nbsp;&nbsp;&nbsp;{<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"formula": "YBa2Cu3O7",<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"predicted_tc": "92K",<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"confidence": 0.94,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"synthesis_route": "solid-state reaction"<br>
&nbsp;&nbsp;&nbsp;&nbsp;},<br>
&nbsp;&nbsp;&nbsp;&nbsp;...<br>
&nbsp;&nbsp;]<br>
}
            </div>
        </div>
    </section>

    <footer class="footer">
        <p>© 2024 ORION Platform. Built with ❤️ by the ORION Team</p>
        <p style="margin-top: 1rem; opacity: 0.8;">
            <a href="https://github.com/orion-platform" style="color: white; text-decoration: none;">GitHub</a> • 
            <a href="https://docs.orion-platform.ai" style="color: white; text-decoration: none;">Documentation</a> • 
            <a href="mailto:support@orion-platform.ai" style="color: white; text-decoration: none;">Support</a>
        </p>
    </footer>
</body>
</html>
"""

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        else:
            super().do_GET()

def run_demo(port=8080):
    """Run the demo server"""
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║     🚀 ORION Platform Demo Server                   ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    
    ✅ Server is running!
    
    📱 Access the demo at:
    
       http://localhost:{port}
    
    Press Ctrl+C to stop the server.
    """)
    
    with socketserver.TCPServer(("", port), DemoHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    run_demo()