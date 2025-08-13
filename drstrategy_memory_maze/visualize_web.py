#!/usr/bin/env python3
"""Web-based visualization for DrStrategy Memory Maze environments.

This script creates a simple web server that streams visualization data
in real-time, suitable for headless environments and remote access.

Usage:
    python visualize_web.py [--env ENV_ID] [--port PORT] [--host HOST]

Examples:
    python visualize_web.py --env DrStrategy-MemoryMaze-4x7x7-v0 --port 8080
    python visualize_web.py --host 0.0.0.0 --port 8000  # Allow external access
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import base64
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

import drstrategy_memory_maze


class EnvironmentRunner:
    """Runs environment and collects data for web visualization."""
    
    def __init__(self, env_id: str = 'DrStrategy-MemoryMaze-4x7x7-v0'):
        """Initialize the environment runner."""
        self.env_id = env_id
        self.env = gym.make(env_id)
        
        # State tracking
        self.current_obs: Optional[Dict[str, np.ndarray]] = None
        self.current_info: Optional[Dict[str, Any]] = None
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0
        self.recent_actions = []
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Initialize
        self._reset_environment()
        
        print(f"Environment created: {env_id}")
        print(f"Action space: {self.env.action_space}")

    def _reset_environment(self) -> None:
        """Reset the environment."""
        self.current_obs, self.current_info = self.env.reset()
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_count += 1

    def _take_random_action(self) -> None:
        """Take a random action and update state."""
        if self.paused:
            return
            
        action = self.env.action_space.sample()
        
        # Track recent actions
        if hasattr(self.env.action_space, 'n'):
            self.recent_actions.append(action)
            if len(self.recent_actions) > 20:
                self.recent_actions.pop(0)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update state
        self.current_obs = obs
        self.current_info = info
        self.episode_reward += reward
        self.episode_step += 1
        self.total_steps += 1
        
        # Handle episode end
        if terminated or truncated:
            print(f"Episode {self.episode_count} finished: {self.episode_step} steps, reward={self.episode_reward:.3f}")
            self._reset_environment()

    def get_state_data(self) -> Dict[str, Any]:
        """Get current state data for web interface."""
        if self.current_obs is None:
            return {}
            
        # Convert image to base64 for web transfer
        image = self.current_obs.get('image', np.zeros((64, 64, 3), dtype=np.uint8))
        
        # Create matplotlib figure for image
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        step_count_raw = self.current_obs.get("step_count", [0])[0]
        ax.set_title(f'Agent View (Step {int(step_count_raw)})')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        # Get target color and convert to native Python types
        target_color = self.current_obs.get('target_color', np.array([1.0, 0.0, 0.0]))
        target_color_list = [float(x) for x in target_color]  # Convert to native float
        
        # Action histogram data
        action_counts = []
        if hasattr(self.env.action_space, 'n') and self.recent_actions:
            action_counts_np = np.bincount(self.recent_actions, minlength=self.env.action_space.n)
            action_counts = [int(x) for x in action_counts_np]  # Convert to native int
        
        # Get step count and convert to native int
        step_count = int(self.current_obs.get('step_count', [0])[0])
        max_steps = self.current_info.get('max_steps', 'N/A')
        if isinstance(max_steps, np.integer):
            max_steps = int(max_steps)
        
        return {
            'env_id': str(self.env_id),
            'episode': int(self.episode_count),
            'episode_step': int(self.episode_step),
            'total_steps': int(self.total_steps),
            'episode_reward': float(self.episode_reward),
            'step_count': step_count,
            'max_steps': max_steps,
            'target_color': target_color_list,
            'target_color_hex': f"#{int(target_color_list[0]*255):02x}{int(target_color_list[1]*255):02x}{int(target_color_list[2]*255):02x}",
            'action_counts': action_counts,
            'action_space_size': int(self.env.action_space.n) if hasattr(self.env.action_space, 'n') else 0,
            'image_base64': str(img_base64),
            'running': bool(self.running),
            'paused': bool(self.paused)
        }

    def start(self) -> None:
        """Start the environment runner loop."""
        self.running = True
        
        def run_loop():
            while self.running:
                self._take_random_action()
                time.sleep(0.05)  # 20 FPS
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the environment runner."""
        self.running = False
        self.env.close()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused


class WebVisualizationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for web visualization."""
    
    def __init__(self, *args, env_runner=None, **kwargs):
        self.env_runner = env_runner
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/':
            self._serve_main_page()
        elif parsed_path.path == '/api/state':
            self._serve_state_api()
        elif parsed_path.path == '/api/control':
            self._handle_control_api(parsed_path.query)
        else:
            self._serve_404()

    def _serve_main_page(self):
        """Serve the main HTML page."""
        html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>DrStrategy Memory Maze Visualization</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .stats { font-family: monospace; line-height: 1.6; }
        .target-color { width: 100px; height: 100px; border: 3px solid black; margin: 10px auto; border-radius: 10px; }
        .controls { text-align: center; margin: 20px 0; }
        .controls button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .pause-btn { background-color: #f39c12; color: white; }
        .restart-btn { background-color: #e74c3c; color: white; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; text-align: center; font-weight: bold; }
        .running { background-color: #2ecc71; color: white; }
        .paused { background-color: #f39c12; color: white; }
        .action-bar { height: 20px; background-color: #3498db; margin: 2px 0; border-radius: 2px; }
        img { max-width: 100%; border-radius: 5px; }
        .refresh-rate { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DrStrategy Memory Maze Visualization</h1>
            <div id="status" class="status">Loading...</div>
            <div class="controls">
                <button class="pause-btn" onclick="togglePause()">⏸️ Pause/Resume</button>
                <button class="restart-btn" onclick="restart()">🔄 Restart Episode</button>
            </div>
            <div class="refresh-rate">Updates every 50ms</div>
        </div>
        
        <div class="dashboard">
            <div class="panel">
                <h3>Agent Observation</h3>
                <div style="text-align: center;">
                    <img id="agent-image" src="" alt="Agent view loading..." />
                </div>
            </div>
            
            <div class="panel">
                <h3>Target Color</h3>
                <div id="target-color" class="target-color"></div>
                <div id="target-rgb" style="text-align: center; font-family: monospace;"></div>
            </div>
            
            <div class="panel">
                <h3>Episode Statistics</h3>
                <div id="stats" class="stats">Loading...</div>
            </div>
            
            <div class="panel">
                <h3>Action History</h3>
                <div id="action-history">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        function updateVisualization() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    if (data.paused) {
                        statusEl.textContent = '⏸️ PAUSED';
                        statusEl.className = 'status paused';
                    } else if (data.running) {
                        statusEl.textContent = '▶️ RUNNING';
                        statusEl.className = 'status running';
                    }
                    
                    // Update agent image
                    if (data.image_base64) {
                        document.getElementById('agent-image').src = 'data:image/png;base64,' + data.image_base64;
                    }
                    
                    // Update target color
                    document.getElementById('target-color').style.backgroundColor = data.target_color_hex;
                    document.getElementById('target-rgb').textContent = 
                        `RGB: (${data.target_color[0].toFixed(2)}, ${data.target_color[1].toFixed(2)}, ${data.target_color[2].toFixed(2)})`;
                    
                    // Update stats
                    const stats = `
Environment: ${data.env_id}
Episode: ${data.episode}
Episode Step: ${data.episode_step}
Total Steps: ${data.total_steps}
Episode Reward: ${data.episode_reward}
Step Count: ${data.step_count}
Max Steps: ${data.max_steps}
                    `.trim();
                    document.getElementById('stats').textContent = stats;
                    
                    // Update action history
                    const actionHistoryEl = document.getElementById('action-history');
                    if (data.action_counts && data.action_counts.length > 0) {
                        const maxCount = Math.max(...data.action_counts);
                        let historyHtml = '';
                        for (let i = 0; i < data.action_counts.length; i++) {
                            const count = data.action_counts[i];
                            const width = maxCount > 0 ? (count / maxCount) * 100 : 0;
                            historyHtml += `
                                <div>Action ${i}: ${count}</div>
                                <div class="action-bar" style="width: ${width}%;"></div>
                            `;
                        }
                        actionHistoryEl.innerHTML = historyHtml;
                    } else {
                        actionHistoryEl.innerHTML = '<div>No action data available</div>';
                    }
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    document.getElementById('status').textContent = '❌ CONNECTION ERROR';
                    document.getElementById('status').className = 'status';
                });
        }
        
        function togglePause() {
            fetch('/api/control?action=pause')
                .then(() => updateVisualization())
                .catch(error => console.error('Control error:', error));
        }
        
        function restart() {
            fetch('/api/control?action=restart')
                .then(() => updateVisualization())
                .catch(error => console.error('Control error:', error));
        }
        
        // Start updating
        updateVisualization();
        setInterval(updateVisualization, 50); // 20 FPS
    </script>
</body>
</html>
        '''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def _serve_state_api(self):
        """Serve the state API endpoint."""
        if self.env_runner:
            state_data = self.env_runner.get_state_data()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(state_data).encode())
        else:
            self._serve_404()

    def _handle_control_api(self, query_string):
        """Handle control API requests."""
        params = urllib.parse.parse_qs(query_string)
        action = params.get('action', [''])[0]
        
        if self.env_runner:
            if action == 'pause':
                self.env_runner.toggle_pause()
            elif action == 'restart':
                self.env_runner._reset_environment()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
        else:
            self._serve_404()

    def _serve_404(self):
        """Serve 404 page."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 Not Found</h1>')

    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass


def run_web_server(env_id: str, host: str = 'localhost', port: int = 8080):
    """Run the web visualization server."""
    # Create environment runner
    env_runner = EnvironmentRunner(env_id)
    env_runner.start()
    
    # Create handler with environment runner
    def handler(*args, **kwargs):
        WebVisualizationHandler(*args, env_runner=env_runner, **kwargs)
    
    # Start web server
    server = HTTPServer((host, port), handler)
    
    print(f"Web visualization server starting...")
    print(f"Environment: {env_id}")
    print(f"Server: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        env_runner.stop()
        server.shutdown()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Web-based visualization of DrStrategy Memory Maze environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    available_envs = [
        'DrStrategy-MemoryMaze-4x7x7-v0',
        'DrStrategy-MemoryMaze-4x15x15-v0', 
        'DrStrategy-MemoryMaze-8x30x30-v0',
        'DrStrategy-MemoryMaze-mzx7x7-v0',
        'DrStrategy-MemoryMaze-mzx15x15-v0'
    ]
    
    parser.add_argument(
        '--env', 
        default='DrStrategy-MemoryMaze-4x7x7-v0',
        choices=available_envs,
        help='Environment to visualize (default: %(default)s)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind to (default: %(default)s, use 0.0.0.0 for external access)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to serve on (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    run_web_server(args.env, args.host, args.port)


if __name__ == '__main__':
    main()