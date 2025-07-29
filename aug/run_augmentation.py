#!/usr/bin/env python3
"""
Main launcher for the CMCC-34 data augmentation system.
This script provides easy access to the reorganized augmentation tools.
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path

def check_api_server(api_url: str = "http://localhost:8001") -> bool:
    """Check if the API server is running."""
    try:
        # Use /v1/models endpoint since /health doesn't exist on this server
        models_url = f"{api_url.rstrip('/')}/v1/models"
        
        response = requests.get(models_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def launch_api_server() -> subprocess.Popen:
    """Launch the GLM-4 API server in the background."""
    try:
        aug_dir = Path(__file__).parent
        api_server_path = aug_dir / "../inference/glm4v_server.py"
        
        print("🚀 Starting GLM-4 API server...")
        print(f"   Command: python {api_server_path}")
        print("   This may take 30-60 seconds to load the model...")
        
        # Launch server in background
        process = subprocess.Popen(
            [sys.executable, str(api_server_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=aug_dir
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start", end="")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            print(".", end="", flush=True)
            
            if check_api_server():
                print("\n✅ API server is now running!")
                return process
        
        print("\n⚠️  Server is still starting. You can check its status manually.")
        return process
        
    except Exception as e:
        print(f"❌ Failed to launch API server: {e}")
        return None

def main():
    """Launch the main augmentation script with server auto-launch capability."""
    # Add src to Python path so we can import modules
    aug_dir = Path(__file__).parent
    src_dir = aug_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Change to aug directory for consistent file operations
    os.chdir(aug_dir)
    
    # Check if API server is running
    server_process = None
    if not check_api_server():
        server_process = launch_api_server()
        if server_process is None:
            print("❌ Could not start API server. Continuing anyway...")
    else:
        print("✅ GLM-4 API server is already running!")
    
    # Import and run the main augmentation script
    try:
        from src.scripts.run_aug import main as run_aug_main
        
        print("\n" + "="*60)
        print("🚀 Starting CMCC-34 Data Augmentation")
        print("="*60)
        
        run_aug_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this script from the aug directory")
        if server_process:
            print("🛑 Stopping API server...")
            server_process.terminate()
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Process interrupted by user")
        if server_process:
            print("🛑 Stopping API server...")
            server_process.terminate()
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error running augmentation: {e}")
        if server_process:
            print("🛑 Stopping API server...")
            server_process.terminate()
        sys.exit(1)
    
    finally:
        # Ask user if they want to keep the server running
        if server_process and server_process.poll() is None:
            print("\n" + "="*60)
            print("🏁 Augmentation Complete!")
            print("="*60)
            keep_running = input("Keep the GLM-4 API server running? [Y/n]: ").strip().lower()
            
            if keep_running not in ['y', 'yes', '']:
                print("🛑 Stopping API server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                print("✅ API server stopped.")
            else:
                print("✅ API server continues running in the background.")
                print("   You can stop it later by pressing Ctrl+C in its terminal")
                print(f"   or by killing process PID: {server_process.pid}")

if __name__ == "__main__":
    main()