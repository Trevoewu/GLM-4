#!/usr/bin/env python3
"""
Script to install evaluation dependencies for visualization.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install evaluation dependencies."""
    print("Installing evaluation dependencies for visualization...")
    
    # Core visualization packages
    packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("✅ All visualization dependencies installed. You can now run evaluation with confusion matrix plots.")
    else:
        print("⚠️  Some packages failed to install. Confusion matrix plots may not work.")

if __name__ == "__main__":
    main()