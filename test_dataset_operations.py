#!/usr/bin/env python3
"""
Test Dataset Operations with Ray
This script tests dataset operations to ensure they work with the new timeout settings.
"""

import os
import sys
import time
import pandas as pd
from google.cloud import aiplatform

# Set environment variables for better Ray stability
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "60"
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "60" 
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "120"
os.environ["RAY_gcs_rpc_server_retry_timeout_s"] = "10"
os.environ["RAY_gcs_rpc_server_retry_interval_s"] = "1"
os.environ["RAY_gcs_rpc_server_max_retries"] = "5"
os.environ["RAY_DATA_READ_TIMEOUT_S"] = "60"
os.environ["RAY_DATA_WRITE_TIMEOUT_S"] = "60"

def test_dataset_operations():
    """Test basic dataset operations with retry logic."""
    print("🔧 Testing dataset operations with improved timeouts...")
    
    try:
        import ray
        
        # Initialize Vertex AI
        aiplatform.init(project='fashion-app-f2861', location='us-west2')
        print("✅ Vertex AI initialized successfully")
        
        # Connect to Ray cluster
        ray_address = "vertex_ray://projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801"
        
        print(f"🔄 Connecting to Ray cluster: {ray_address}")
        
        ray.init(
            address=ray_address,
            log_to_driver=True,
            ignore_reinit_error=True
        )
        
        print("✅ Ray connection successful!")
        
        # Test 1: Create a simple dataset
        print("\n📊 Test 1: Creating a simple dataset...")
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Create a simple test dataset
                test_data = pd.DataFrame({
                    'id': range(100),
                    'value': [f'item_{i}' for i in range(100)]
                })
                
                ds = ray.data.from_pandas(test_data)
                print("✅ Dataset created successfully")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} to create dataset failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ Failed to create dataset after all attempts")
                    return False
        
        # Test 2: Get dataset size
        print("\n📊 Test 2: Getting dataset size...")
        for attempt in range(max_retries):
            try:
                size = ds.count()
                print(f"✅ Dataset size: {size}")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} to get dataset size failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ Failed to get dataset size after all attempts")
                    return False
        
        # Test 3: Split dataset
        print("\n📊 Test 3: Splitting dataset...")
        for attempt in range(max_retries):
            try:
                train_ds, val_ds = ds.split_proportionately([0.8])
                print("✅ Dataset split successful")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} to split dataset failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ Failed to split dataset after all attempts")
                    return False
        
        # Test 4: Apply a simple transformation
        print("\n📊 Test 4: Applying transformation...")
        for attempt in range(max_retries):
            try:
                def add_prefix(batch):
                    batch['prefixed_value'] = 'test_' + batch['value']
                    return batch
                
                transformed_ds = ds.map_batches(add_prefix, batch_format="pandas", batch_size=10)
                print("✅ Transformation applied successfully")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} to apply transformation failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ Failed to apply transformation after all attempts")
                    return False
        
        # Test 5: Limit dataset
        print("\n📊 Test 5: Limiting dataset...")
        for attempt in range(max_retries):
            try:
                limited_ds = ds.limit(50)
                limited_size = limited_ds.count()
                print(f"✅ Dataset limited successfully: {limited_size} rows")
                break
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} to limit dataset failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("❌ Failed to limit dataset after all attempts")
                    return False
        
        ray.shutdown()
        print("\n🎉 All dataset operations tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset operations test failed: {e}")
        print("\n💡 Troubleshooting suggestions:")
        print("   1. Check if your Vertex AI Ray cluster is running")
        print("   2. Verify your Google Cloud authentication")
        print("   3. Check network connectivity")
        print("   4. Try restarting the Ray cluster")
        return False

if __name__ == "__main__":
    success = test_dataset_operations()
    if success:
        print("\n🎉 Ready to run the full model training!")
        print("💡 Run: python model.py")
    else:
        print("\n❌ Please fix the dataset operation issues before running the model.")
        sys.exit(1) 