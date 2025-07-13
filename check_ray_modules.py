#!/usr/bin/env python3
"""
Check available Ray modules and connection options
"""

import ray
import sys
import os

def check_ray_modules():
    """Check what Ray modules are available."""
    print("🔍 Checking Ray modules and capabilities...")
    print(f"Ray version: {ray.__version__}")
    
    # Check if vertex_ray module exists
    try:
        import vertex_ray
        print("✅ vertex_ray module found")
    except ImportError:
        print("❌ vertex_ray module not found")
    
    # Check Ray client capabilities
    try:
        from ray.util.client import ray as ray_client
        print("✅ Ray client module available")
    except ImportError:
        print("❌ Ray client module not available")
    
    # List available Ray modules
    ray_modules = [name for name in dir(ray) if not name.startswith('_')]
    print(f"\nAvailable Ray modules: {len(ray_modules)}")
    for module in sorted(ray_modules)[:20]:  # Show first 20
        print(f"  - {module}")
    
    # Check if we can import google.cloud.aiplatform
    try:
        from google.cloud import aiplatform
        print(f"\n✅ google.cloud.aiplatform available (version: {aiplatform.__version__})")
        
        # Check for vertex_ray in aiplatform
        aiplatform_modules = [name for name in dir(aiplatform) if not name.startswith('_')]
        if 'vertex_ray' in aiplatform_modules:
            print("✅ vertex_ray found in aiplatform")
        else:
            print("❌ vertex_ray not found in aiplatform")
            
    except ImportError as e:
        print(f"❌ google.cloud.aiplatform not available: {e}")

def test_alternative_connection():
    """Test alternative connection methods."""
    print("\n🔧 Testing alternative connection methods...")
    
    # Method 1: Try direct Ray client connection
    try:
        print("Testing direct Ray client connection...")
        # This would need the actual cluster endpoint
        pass
    except Exception as e:
        print(f"Direct connection failed: {e}")
    
    # Method 2: Check if we can use Ray job submission
    try:
        from ray.job_submission import JobSubmissionClient
        print("✅ Ray job submission client available")
    except ImportError:
        print("❌ Ray job submission client not available")

if __name__ == "__main__":
    check_ray_modules()
    test_alternative_connection() 