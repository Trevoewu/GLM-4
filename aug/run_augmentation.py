#!/usr/bin/env python3
"""
Main launcher for the CMCC-34 data augmentation system.
This script provides easy access to the reorganized augmentation tools.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the main augmentation script."""
    # Add src to Python path so we can import modules
    aug_dir = Path(__file__).parent
    src_dir = aug_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Change to aug directory for consistent file operations
    os.chdir(aug_dir)
    
    # Import and run the main augmentation script
    try:
        from src.scripts.run_aug import main as run_aug_main
        run_aug_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this script from the aug directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error running augmentation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()