#!/usr/bin/env python3
"""
Ray Cluster Diagnostics Script
This script helps diagnose and troubleshoot Ray cluster connection issues.
"""

import os
import sys
import time
import subprocess
import json
from google.cloud import aiplatform

def set_ray_environment_variables():
    """Set environment variables for better Ray stability."""
    env_vars = {
        "RAY_gcs_rpc_server_reconnect_timeout_s": "30",
        "RAY_gcs_rpc_server_connect_timeout_s": "30", 
        "RAY_gcs_rpc_server_request_timeout_s": "60",
        "RAY_gcs_rpc_server_retry_timeout_s": "10",
        "RAY_gcs_rpc_server_retry_interval_s": "1",
        "RAY_gcs_rpc_server_max_retries": "5",
        "RAY_gcs_rpc_server_retry_interval_s": "1",
        "RAY_gcs_rpc_server_max_retries": "5"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"🔧 Set {key} = {value}")

def check_vertex_ai_cluster():
    """Check Vertex AI Ray cluster status."""
    print("\n🔍 Checking Vertex AI Ray cluster status...")
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project='fashion-app-f2861', location='us-west2')
        print("✅ Vertex AI initialized successfully")
        
        # List persistent resources (Ray clusters)
        client = aiplatform.gapic.PersistentResourceServiceClient()
        parent = f"projects/927709385665/locations/us-west2"
        
        request = aiplatform.gapic.ListPersistentResourcesRequest(parent=parent)
        page_result = client.list_persistent_resources(request=request)
        
        clusters = []
        for response in page_result:
            clusters.append(response)
        
        if clusters:
            print(f"📊 Found {len(clusters)} persistent resources:")
            for cluster in clusters:
                print(f"   - Name: {cluster.name}")
                print(f"     State: {cluster.state}")
                print(f"     Create Time: {cluster.create_time}")
                print(f"     Update Time: {cluster.update_time}")
                print()
        else:
            print("❌ No persistent resources found")
            
    except Exception as e:
        print(f"❌ Error checking Vertex AI cluster: {e}")

def test_ray_connection():
    """Test Ray connection with detailed diagnostics."""
    print("\n🔍 Testing Ray connection...")
    
    try:
        import ray
        
        # Try to connect to the cluster
        ray_address = "vertex_ray://projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801"
        
        print(f"🔄 Attempting to connect to: {ray_address}")
        
        ray.init(
            address=ray_address,
            log_to_driver=True,
            ignore_reinit_error=True
        )
        
        print("✅ Ray connection successful!")
        
        # Get cluster information
        try:
            resources = ray.cluster_resources()
            print(f"📊 Cluster resources: {resources}")
        except Exception as e:
            print(f"⚠️  Could not get cluster resources: {e}")
        
        try:
            nodes = ray.nodes()
            print(f"🖥️  Cluster nodes: {len(nodes)}")
            for node in nodes:
                print(f"   - Node: {node.get('NodeID', 'Unknown')}")
                print(f"     State: {node.get('Alive', False)}")
                print(f"     Resources: {node.get('Resources', {})}")
        except Exception as e:
            print(f"⚠️  Could not get cluster nodes: {e}")
        
        # Test basic Ray functionality
        try:
            @ray.remote
            def test_function():
                return "Hello from Ray!"
            
            result = ray.get(test_function.remote())
            print(f"✅ Ray remote function test: {result}")
        except Exception as e:
            print(f"❌ Ray remote function test failed: {e}")
        
        ray.shutdown()
        
    except Exception as e:
        print(f"❌ Ray connection failed: {e}")
        print("💡 This could be due to:")
        print("   - Cluster is not running")
        print("   - Network connectivity issues")
        print("   - Authentication problems")
        print("   - GCS server issues")

def check_network_connectivity():
    """Check basic network connectivity."""
    print("\n🌐 Checking network connectivity...")
    
    # Test basic internet connectivity
    try:
        import urllib.request
        urllib.request.urlopen('http://www.google.com', timeout=10)
        print("✅ Internet connectivity: OK")
    except Exception as e:
        print(f"❌ Internet connectivity failed: {e}")
    
    # Test Google Cloud connectivity
    try:
        import urllib.request
        urllib.request.urlopen('https://cloud.google.com', timeout=10)
        print("✅ Google Cloud connectivity: OK")
    except Exception as e:
        print(f"❌ Google Cloud connectivity failed: {e}")

def check_ray_installation():
    """Check Ray installation and version."""
    print("\n📦 Checking Ray installation...")
    
    try:
        import ray
        print(f"✅ Ray version: {ray.__version__}")
    except ImportError:
        print("❌ Ray not installed")
        return False
    
    try:
        import ray.air
        print("✅ Ray AIR available")
    except ImportError:
        print("❌ Ray AIR not available")
    
    try:
        import ray.data
        print("✅ Ray Data available")
    except ImportError:
        print("❌ Ray Data not available")
    
    try:
        import ray.train
        print("✅ Ray Train available")
    except ImportError:
        print("❌ Ray Train not available")
    
    return True

def main():
    """Main diagnostic function."""
    print("🔧 Ray Cluster Diagnostics")
    print("=" * 50)
    
    # Set environment variables
    set_ray_environment_variables()
    
    # Check Ray installation
    if not check_ray_installation():
        print("❌ Please install Ray first: pip install ray[air]")
        return
    
    # Check network connectivity
    check_network_connectivity()
    
    # Check Vertex AI cluster
    check_vertex_ai_cluster()
    
    # Test Ray connection
    test_ray_connection()
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostics complete!")
    print("\n💡 If you're still having issues:")
    print("   1. Check if your Vertex AI Ray cluster is running")
    print("   2. Verify your Google Cloud authentication")
    print("   3. Check network firewall settings")
    print("   4. Try restarting the Ray cluster")
    print("   5. Consider using local Ray as fallback")

if __name__ == "__main__":
    main() 