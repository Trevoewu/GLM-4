#!/usr/bin/env python3
"""
Script to install Chinese fonts for matplotlib visualization.
This script helps resolve font rendering issues for Chinese characters in confusion matrix plots.
"""

import os
import subprocess
import sys
import platform

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_fonts_ubuntu_debian():
    """Install Chinese fonts on Ubuntu/Debian systems."""
    print("Installing Chinese fonts for Ubuntu/Debian...")
    
    commands = [
        ("apt-get update", "Update package list"),
        ("apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei", "Install WenQuanYi fonts"),
        ("apt-get install -y fonts-noto-cjk", "Install Noto CJK fonts"),
        ("fc-cache -fv", "Update font cache")
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    return success_count == len(commands)

def install_fonts_centos_rhel():
    """Install Chinese fonts on CentOS/RHEL systems."""
    print("Installing Chinese fonts for CentOS/RHEL...")
    
    commands = [
        ("yum install -y wqy-microhei-fonts wqy-zenhei-fonts", "Install WenQuanYi fonts"),
        ("yum install -y google-noto-cjk-fonts", "Install Noto CJK fonts"),
        ("fc-cache -fv", "Update font cache")
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    return success_count == len(commands)

def install_fonts_arch():
    """Install Chinese fonts on Arch Linux."""
    print("Installing Chinese fonts for Arch Linux...")
    
    commands = [
        ("pacman -S --noconfirm wqy-microhei wqy-zenhei", "Install WenQuanYi fonts"),
        ("pacman -S --noconfirm noto-fonts-cjk", "Install Noto CJK fonts"),
        ("fc-cache -fv", "Update font cache")
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    return success_count == len(commands)

def test_font_availability():
    """Test if Chinese fonts are available after installation."""
    try:
        import matplotlib.font_manager as fm
        
        # Check for Chinese fonts
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        found_fonts = []
        for font in chinese_fonts:
            if font in available_fonts:
                found_fonts.append(font)
        
        if found_fonts:
            print(f"✅ Chinese fonts found: {', '.join(found_fonts)}")
            return True
        else:
            print("❌ No Chinese fonts found after installation")
            return False
            
    except ImportError:
        print("❌ matplotlib not available for font testing")
        return False

def test_chinese_rendering():
    """Test if Chinese characters render correctly."""
    try:
        import matplotlib.pyplot as plt
        
        # Configure matplotlib
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'sans-serif']
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
        
        print(f"✅ Chinese font rendering test successful. Check {test_file}")
        return True
        
    except Exception as e:
        print(f"❌ Chinese font rendering test failed: {e}")
        return False

def main():
    """Main function to install Chinese fonts."""
    print("🔤 Chinese Font Installation Script")
    print("=" * 50)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("⚠️  Warning: This script may need root privileges to install system fonts.")
        print("   If installation fails, try running with sudo.")
        print()
    
    # Detect OS
    system = platform.system().lower()
    if system == "linux":
        # Try to detect distribution
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read().lower()
                if 'ubuntu' in content or 'debian' in content:
                    success = install_fonts_ubuntu_debian()
                elif 'centos' in content or 'rhel' in content or 'redhat' in content:
                    success = install_fonts_centos_rhel()
                elif 'arch' in content:
                    success = install_fonts_arch()
                else:
                    print("❌ Unsupported Linux distribution")
                    print("Please install Chinese fonts manually:")
                    print("- WenQuanYi Micro Hei")
                    print("- Noto Sans CJK SC")
                    print("- Or any other Chinese font")
                    return False
        except FileNotFoundError:
            print("❌ Could not detect Linux distribution")
            return False
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False
    
    if success:
        print("\n🔄 Testing font availability...")
        if test_font_availability():
            print("\n🔄 Testing Chinese character rendering...")
            if test_chinese_rendering():
                print("\n✅ Chinese font installation completed successfully!")
                print("   You can now run evaluation scripts without font warnings.")
            else:
                print("\n⚠️  Font installation completed but rendering test failed.")
                print("   You may need to restart your Python environment.")
        else:
            print("\n⚠️  Font installation completed but availability test failed.")
            print("   You may need to restart your Python environment.")
    else:
        print("\n❌ Font installation failed.")
        print("   Please install Chinese fonts manually or run with sudo.")
    
    return success

if __name__ == "__main__":
    main()