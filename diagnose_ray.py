#!/usr/bin/env python3
"""
Ray Environment Diagnostic Script
Checks your local Ray setup and provides troubleshooting information.
"""

import sys
import os
import subprocess

def check_package(package_name):
    """Check if a package is installed and return its version."""
    try:
        import importlib
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔍 Ray Environment Diagnostic")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check Ray installation
    ray_installed, ray_version = check_package('ray')
    print(f"Ray installed: {ray_installed}")
    if ray_installed:
        print(f"Ray version: {ray_version}")
    
    # Check Google Cloud AI Platform
    aiplatform_installed, aiplatform_version = check_package('google.cloud.aiplatform')
    print(f"Google Cloud AI Platform installed: {aiplatform_installed}")
    if aiplatform_installed:
        print(f"AI Platform version: {aiplatform_version}")
    
    # Check other dependencies
    packages = ['torch', 'pandas', 'numpy', 'PIL']
    for package in packages:
        installed, version = check_package(package)
        print(f"{package} installed: {installed} (version: {version})")
    
    print("\n🔧 Environment Variables:")
    ray_vars = ['RAY_IGNORE_VERSION_MISMATCH', 'RAY_DISABLE_VERSION_CHECK', 'GOOGLE_APPLICATION_CREDENTIALS']
    for var in ray_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    print("\n🌐 Network and Authentication:")
    
    # Check gcloud authentication
    success, stdout, stderr = run_command("gcloud auth list")
    if success:
        print("✅ gcloud authentication configured")
        print(f"   Active accounts: {stdout.strip()}")
    else:
        print("❌ gcloud authentication issue")
        print(f"   Error: {stderr}")
    
    # Check cluster status
    success, stdout, stderr = run_command("gcloud ai persistent-resources list --region=us-west2")
    if success:
        print("✅ Can access Vertex AI resources")
        if "cluster-20250712-190801" in stdout:
            print("✅ Target cluster found")
        else:
            print("❌ Target cluster not found in list")
    else:
        print("❌ Cannot access Vertex AI resources")
        print(f"   Error: {stderr}")
    
    print("\n💡 Recommendations:")
    if not ray_installed:
        print("1. Install Ray: pip install ray[default]==2.9.0")
    elif ray_version != "2.9.0":
        print(f"1. Update Ray to version 2.9.0: pip install ray[default]==2.9.0")
    
    if not aiplatform_installed:
        print("2. Install Google Cloud AI Platform: pip install google-cloud-aiplatform")
    
    print("3. Run the test script: python raytest.py")

if __name__ == "__main__":
    main() 