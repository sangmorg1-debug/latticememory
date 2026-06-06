"""Utility script to build LatticeMemory source and binary distributions."""
import subprocess
import sys
import shutil
from pathlib import Path

def clean_build_dirs():
    """Remove old build artifacts to ensure a clean release."""
    print("Cleaning build, dist, and egg-info directories...")
    for folder in ['build', 'dist']:
        path = Path(folder)
        if path.exists():
            shutil.rmtree(path)
    for path in Path('.').glob('*.egg-info'):
        shutil.rmtree(path)

def build_packages():
    """Run python -m build to create sdist and wheel."""
    print("Building sdist and wheel...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to pre-install build/twine. Proceeding anyway.")
    
    try:
        subprocess.run([sys.executable, "-m", "build"], check=True)
        print("Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Build failed: {e}")
        sys.exit(1)

def verify_packages():
    """Run twine check on generated packages."""
    print("Checking packages with twine...")
    try:
        subprocess.run("twine check dist/*", shell=True, check=True)
        print("Twine check passed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Twine check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clean_build_dirs()
    build_packages()
    verify_packages()
    print("\nPackage is ready for PyPI upload.")
    print("To upload, run: python -m twine upload dist/*")
