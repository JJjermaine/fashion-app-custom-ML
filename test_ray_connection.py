#!/usr/bin/env python3
"""
Simple Ray Connection Test
This script tests the Ray connection before running the full model training.
"""

import os
import sys
import time

# Set environment variables for better Ray stability
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "30"
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "30" 
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "60"
os.environ["RAY_gcs_rpc_server_retry_timeout_s"] = "10"
os.environ["RAY_gcs_rpc_server_retry_interval_s"] = "1"
os.environ["RAY_gcs_rpc_server_max_retries"] = "5"

def test_ray_connection():
    """Test Ray connection with the improved setup."""
    print("🔧 Testing Ray connection with improved timeouts...")
    
    try:
        from google.cloud import aiplatform
        import ray
        
        # Initialize Vertex AI
        aiplatform.init(project='fashion-app-f2861', location='us-west2')
        print("✅ Vertex AI initialized successfully")
        
        # Try to connect to the Ray cluster
        ray_address = "vertex_ray://projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801"
        
        print(f"🔄 Attempting to connect to: {ray_address}")
        
        ray.init(
            address=ray_address,
            log_to_driver=True,
            ignore_reinit_error=True
        )
        
        print("✅ Ray connection successful!")
        
        # Test basic functionality
        @ray.remote
        def simple_test():
            return "Connection test successful!"
        
        result = ray.get(simple_test.remote())
        print(f"✅ Remote function test: {result}")
        
        # Get cluster info
        try:
            resources = ray.cluster_resources()
            print(f"📊 Available resources: {resources}")
        except Exception as e:
            print(f"⚠️  Could not get cluster resources: {e}")
        
        ray.shutdown()
        print("✅ Ray connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Ray connection test failed: {e}")
        print("\n💡 Troubleshooting suggestions:")
        print("   1. Check if your Vertex AI Ray cluster is running")
        print("   2. Verify your Google Cloud authentication")
        print("   3. Run the diagnostic script: python ray_diagnostics.py")
        print("   4. Try restarting the Ray cluster")
        return False

if __name__ == "__main__":
    success = test_ray_connection()
    if success:
        print("\n🎉 Ready to run the full model training!")
        print("💡 Run: python model.py")
    else:
        print("\n❌ Please fix the connection issues before running the model.")
        sys.exit(1) 