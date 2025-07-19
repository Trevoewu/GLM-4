#!/usr/bin/env python3
"""
Script to install Chinese fonts for matplotlib visualization.
"""

import subprocess
import sys
import os

def install_fonts():
    """Install Chinese fonts for better visualization."""
    print("Installing Chinese fonts for matplotlib...")
    
    # Check if we're on Ubuntu/Debian
    try:
        # Install fonts-noto-cjk for Chinese characters
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "fonts-noto-cjk"], check=True, capture_output=True)
        print("✅ Successfully installed Noto CJK fonts")
        
        # Clear matplotlib font cache
        import matplotlib.font_manager as fm
        fm._rebuild()
        print("✅ Cleared matplotlib font cache")
        
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install fonts via apt-get")
        return False
    except ImportError:
        print("❌ matplotlib not available")
        return False

def test_chinese_fonts():
    """Test if Chinese fonts are working."""
    try:
        import matplotlib.pyplot as plt
        
        # Configure matplotlib
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create a simple test plot
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title('测试中文标题 Test Chinese Title')
        plt.xlabel('咨询（含查询）业务规定')
        plt.ylabel('办理取消')
        
        # Save test plot
        test_file = "chinese_font_test.png"
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Chinese font test successful. Check {test_file}")
        return True
        
    except Exception as e:
        print(f"❌ Chinese font test failed: {e}")
        return False

def main():
    """Main function."""
    print("Setting up Chinese fonts for matplotlib...")
    
    # Install fonts
    if install_fonts():
        # Test fonts
        if test_chinese_fonts():
            print("✅ Chinese fonts setup completed successfully!")
        else:
            print("⚠️  Font installation completed but test failed.")
    else:
        print("❌ Font installation failed.")
        print("You may need to manually install Chinese fonts or use a different approach.")

if __name__ == "__main__":
    main()