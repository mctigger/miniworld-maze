#!/usr/bin/env python3
"""Web-based visualization for DrStrategy Miniworld environments.

This script creates a simple web server that streams visualization data
in real-time, suitable for headless environments and remote access.

Usage:
    python visualize_web.py [--env ENV_NAME] [--port PORT] [--host HOST]

Examples:
    python visualize_web.py --env PickupObjs --port 8080
    python visualize_web.py --env OneRoom --host 0.0.0.0 --port 8000
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
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

import drstrategy_miniworld
from drstrategy_miniworld.envs import (
    PickupObjs, OneRoom, TwoRoomsVer1, ThreeRooms, 
    RoomObjs, SimToRealGoto, SimToRealPush
)


class EnvironmentRunner:
    """Runs environment and collects data for web visualization."""
    
    def __init__(self, env_name: str = 'OneRoom', **env_kwargs):
        """Initialize the environment runner."""
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        
        # Create environment
        env_classes = {
            'PickupObjs': PickupObjs,
            'OneRoom': OneRoom,
            'TwoRoomsVer1': TwoRoomsVer1,
            'ThreeRooms': ThreeRooms,
            'RoomObjs': RoomObjs,
            'SimToRealGoto': SimToRealGoto,
            'SimToRealPush': SimToRealPush,
        }
        
        if env_name not in env_classes:
            raise ValueError(f"Unknown environment: {env_name}. Available: {list(env_classes.keys())}")
        
        self.env = env_classes[env_name](**env_kwargs)
        
        # State tracking
        self.current_obs: Optional[np.ndarray] = None
        self.current_info: Optional[Dict[str, Any]] = None
        self.episode_reward = 0.0
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0
        self.recent_actions = []
        self.recent_rewards = []
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Initialize
        self._reset_environment()
        
        print(f"Environment created: {env_name} with kwargs: {env_kwargs}")
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
            if len(self.recent_actions) > 50:
                self.recent_actions.pop(0)
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update state
        self.current_obs = obs
        self.current_info = info
        self.episode_reward += reward
        self.episode_step += 1
        self.total_steps += 1
        
        # Track recent rewards
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Handle episode end
        if terminated or truncated:
            print(f"Episode {self.episode_count} finished: {self.episode_step} steps, reward={self.episode_reward:.3f}")
            self._reset_environment()

    def get_state_data(self) -> Dict[str, Any]:
        """Get current state data for web interface."""
        if self.current_obs is None:
            return {}
            
        # Get the observation image
        # Miniworld typically returns RGB images directly
        if isinstance(self.current_obs, np.ndarray):
            image = self.current_obs
        else:
            # If observation is dict-like, try common keys
            image = self.current_obs.get('image', self.current_obs.get('rgb', np.zeros((64, 64, 3), dtype=np.uint8)))
        
        # Ensure image is in correct format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create matplotlib figure for image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title(f'Agent View (Step {self.episode_step})')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        # Action histogram data
        action_counts = []
        if hasattr(self.env.action_space, 'n') and self.recent_actions:
            action_counts_np = np.bincount(self.recent_actions, minlength=self.env.action_space.n)
            action_counts = [int(x) for x in action_counts_np]
        
        # Reward history for plotting
        reward_history = [float(r) for r in self.recent_rewards]
        
        # Get action names if available
        action_names = []
        if hasattr(self.env, 'actions'):
            try:
                for i in range(self.env.action_space.n):
                    action_names.append(f"{i}: {self.env.actions(i).name}")
            except:
                action_names = [f"Action {i}" for i in range(self.env.action_space.n)]
        else:
            action_names = [f"Action {i}" for i in range(getattr(self.env.action_space, 'n', 0))]
        
        # Environment-specific information
        env_info = {}
        if hasattr(self.env, 'num_picked_up'):
            env_info['Objects Picked Up'] = int(self.env.num_picked_up)
        if hasattr(self.env, 'num_objs'):
            env_info['Total Objects'] = int(self.env.num_objs)
        if hasattr(self.env, 'agent') and hasattr(self.env.agent, 'carrying'):
            env_info['Carrying Object'] = bool(self.env.agent.carrying)
        
        return {
            'env_name': str(self.env_name),
            'env_kwargs': dict(self.env_kwargs),
            'episode': int(self.episode_count),
            'episode_step': int(self.episode_step),
            'total_steps': int(self.total_steps),
            'episode_reward': float(self.episode_reward),
            'action_counts': action_counts,
            'action_names': action_names,
            'action_space_size': int(getattr(self.env.action_space, 'n', 0)),
            'reward_history': reward_history,
            'image_base64': str(img_base64),
            'running': bool(self.running),
            'paused': bool(self.paused),
            'env_info': env_info
        }

    def start(self) -> None:
        """Start the environment runner loop."""
        self.running = True
        
        def run_loop():
            while self.running:
                self._take_random_action()
                time.sleep(0.1)  # 10 FPS for 3D environments
        
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
    <title>DrStrategy Miniworld Visualization</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .dashboard { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .bottom-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .stats { font-family: monospace; line-height: 1.6; }
        .controls { text-align: center; margin: 20px 0; }
        .controls button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .pause-btn { background-color: #f39c12; color: white; }
        .restart-btn { background-color: #e74c3c; color: white; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; text-align: center; font-weight: bold; }
        .running { background-color: #2ecc71; color: white; }
        .paused { background-color: #f39c12; color: white; }
        .action-bar { height: 20px; background-color: #3498db; margin: 2px 0; border-radius: 2px; display: flex; align-items: center; padding-left: 5px; color: white; font-size: 12px; }
        .reward-point { width: 3px; height: 100%; background-color: #2ecc71; margin: 1px 0; }
        .reward-point.negative { background-color: #e74c3c; }
        .reward-point.zero { background-color: #95a5a6; }
        .reward-history { height: 100px; border: 1px solid #ddd; position: relative; overflow: hidden; background: #f9f9f9; }
        img { max-width: 100%; border-radius: 5px; }
        .refresh-rate { color: #666; font-size: 12px; }
        .env-info { background: #ecf0f1; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .env-info h4 { margin: 0 0 10px 0; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DrStrategy Miniworld Visualization</h1>
            <div id="status" class="status">Loading...</div>
            <div class="controls">
                <button class="pause-btn" onclick="togglePause()">⏸️ Pause/Resume</button>
                <button class="restart-btn" onclick="restart()">🔄 Restart Episode</button>
            </div>
            <div class="refresh-rate">Updates every 100ms • Random actions</div>
        </div>
        
        <div class="dashboard">
            <div class="panel">
                <h3>Agent 3D View</h3>
                <div style="text-align: center;">
                    <img id="agent-image" src="" alt="Agent view loading..." />
                </div>
            </div>
            
            <div class="panel">
                <h3>Episode Statistics</h3>
                <div id="stats" class="stats">Loading...</div>
                <div id="env-info" class="env-info">
                    <h4>Environment Info</h4>
                    <div id="env-specific">Loading...</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Reward History</h3>
                <div id="reward-history" class="reward-history"></div>
                <div style="font-size: 12px; color: #666; margin-top: 5px;">
                    Last 100 steps • Green: +reward, Red: -reward, Gray: 0
                </div>
            </div>
        </div>
        
        <div class="bottom-panel">
            <div class="panel">
                <h3>Action Distribution</h3>
                <div id="action-history">Loading...</div>
            </div>
            
            <div class="panel">
                <h3>Environment Configuration</h3>
                <div id="env-config" class="stats">Loading...</div>
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
                    
                    // Update stats
                    const stats = `
Environment: ${data.env_name}
Episode: ${data.episode}
Episode Step: ${data.episode_step}
Total Steps: ${data.total_steps}
Episode Reward: ${data.episode_reward.toFixed(3)}
Average Reward: ${data.reward_history.length > 0 ? (data.reward_history.reduce((a,b) => a+b, 0) / data.reward_history.length).toFixed(3) : '0.000'}
                    `.trim();
                    document.getElementById('stats').textContent = stats;
                    
                    // Update environment specific info
                    let envSpecific = '';
                    if (data.env_info) {
                        for (const [key, value] of Object.entries(data.env_info)) {
                            envSpecific += key + ': ' + value + '\\n';
                        }
                    }
                    document.getElementById('env-specific').textContent = envSpecific || 'No additional info';
                    
                    // Update environment configuration
                    let configText = 'Environment: ' + data.env_name + '\\n';
                    if (data.env_kwargs && Object.keys(data.env_kwargs).length > 0) {
                        configText += 'Parameters:\\n';
                        for (const [key, value] of Object.entries(data.env_kwargs)) {
                            configText += '  ' + key + ': ' + value + '\\n';
                        }
                    }
                    document.getElementById('env-config').textContent = configText;
                    
                    // Update action history
                    const actionHistoryEl = document.getElementById('action-history');
                    if (data.action_counts && data.action_counts.length > 0) {
                        const maxCount = Math.max(...data.action_counts);
                        let historyHtml = '';
                        for (let i = 0; i < data.action_counts.length; i++) {
                            const count = data.action_counts[i];
                            const width = maxCount > 0 ? (count / maxCount) * 100 : 0;
                            const actionName = data.action_names[i] || `Action ${i}`;
                            historyHtml += `
                                <div class="action-bar" style="width: ${Math.max(width, 10)}%;">
                                    ${actionName}: ${count}
                                </div>
                            `;
                        }
                        actionHistoryEl.innerHTML = historyHtml;
                    } else {
                        actionHistoryEl.innerHTML = '<div>No action data available</div>';
                    }
                    
                    // Update reward history visualization
                    const rewardHistoryEl = document.getElementById('reward-history');
                    if (data.reward_history && data.reward_history.length > 0) {
                        let historyHtml = '<div style="display: flex; height: 100%; align-items: flex-end;">';
                        for (const reward of data.reward_history) {
                            let className = 'reward-point';
                            if (reward > 0) className += '';
                            else if (reward < 0) className += ' negative';
                            else className += ' zero';
                            
                            const height = Math.max(Math.abs(reward) * 50, 2);
                            historyHtml += `<div class="${className}" style="height: ${height}px;" title="Reward: ${reward}"></div>`;
                        }
                        historyHtml += '</div>';
                        rewardHistoryEl.innerHTML = historyHtml;
                    } else {
                        rewardHistoryEl.innerHTML = '<div>No reward history</div>';
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
        setInterval(updateVisualization, 100); // 10 FPS
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


def run_web_server(env_name: str, host: str = 'localhost', port: int = 8080, **env_kwargs):
    """Run the web visualization server."""
    # Create environment runner
    env_runner = EnvironmentRunner(env_name, **env_kwargs)
    env_runner.start()
    
    # Create handler with environment runner
    def handler(*args, **kwargs):
        WebVisualizationHandler(*args, env_runner=env_runner, **kwargs)
    
    # Start web server
    server = HTTPServer((host, port), handler)
    
    print(f"Web visualization server starting...")
    print(f"Environment: {env_name}")
    print(f"Parameters: {env_kwargs}")
    print(f"Server: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down server...")
        env_runner.stop()
        server.shutdown()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Web-based visualization of DrStrategy Miniworld environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    available_envs = [
        'PickupObjs',
        'OneRoom',
        'TwoRoomsVer1',
        'ThreeRooms',
        'RoomObjs',
        'SimToRealGoto',
        'SimToRealPush'
    ]
    
    parser.add_argument(
        '--env', 
        default='OneRoom',
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
    
    # Environment-specific arguments
    parser.add_argument('--size', type=int, help='Environment size parameter')
    parser.add_argument('--num-objs', type=int, help='Number of objects (for PickupObjs)')
    parser.add_argument('--room-size', type=int, help='Room size (for room navigation envs)')
    parser.add_argument('--door-size', type=int, help='Door size (for room navigation envs)')
    
    args = parser.parse_args()
    
    # Build environment kwargs
    env_kwargs = {}
    if args.size is not None:
        env_kwargs['size'] = args.size
    if args.num_objs is not None:
        env_kwargs['num_objs'] = args.num_objs  
    if args.room_size is not None:
        env_kwargs['room_size'] = args.room_size
    if args.door_size is not None:
        env_kwargs['door_size'] = args.door_size
    
    run_web_server(args.env, args.host, args.port, **env_kwargs)


if __name__ == '__main__':
    main()