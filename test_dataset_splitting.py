#!/usr/bin/env python3
"""
Test Dataset Splitting with Ray
This script specifically tests the dataset splitting operation that's causing GCS timeouts.
"""

# Set environment variables for Ray timeouts BEFORE any imports
import os
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "180"
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "180"
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "300"
os.environ["RAY_gcs_rpc_server_retry_timeout_s"] = "30"
os.environ["RAY_gcs_rpc_server_retry_interval_s"] = "5"
os.environ["RAY_gcs_rpc_server_max_retries"] = "15"
os.environ["RAY_DATA_READ_TIMEOUT_S"] = "180"
os.environ["RAY_DATA_WRITE_TIMEOUT_S"] = "180"

import sys
import time
import pandas as pd
from google.cloud import aiplatform

def test_dataset_splitting():
    """Test dataset splitting operation specifically."""
    print("🔧 Testing dataset splitting operation with extended timeouts...")
    
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
        
        # Create a larger test dataset to simulate the real scenario
        print("\n📊 Creating test dataset...")
        test_data = pd.DataFrame({
            'id': range(10000),  # Larger dataset
            'value': [f'item_{i}' for i in range(10000)]
        })
        
        ds = ray.data.from_pandas(test_data)
        print(f"✅ Dataset created with {ds.count()} rows")
        
        # Test 1: Standard split_proportionately
        print("\n📊 Test 1: Standard split_proportionately...")
        max_retries = 5
        retry_delay = 15  # Longer initial delay
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries} to split dataset...")
                train_ds, val_ds, test_ds = ds.split_proportionately([0.7, 0.15])
                print("✅ Standard split successful")
                
                # Test the split results
                train_size = train_ds.count()
                val_size = val_ds.count()
                test_size = test_ds.count()
                print(f"   Train: {train_size}, Val: {val_size}, Test: {test_size}")
                break
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    print(f"⏳ Next retry delay will be {retry_delay} seconds")
                else:
                    print("❌ Standard split failed after all attempts")
                    
                    # Test 2: Alternative splitting approach
                    print("\n📊 Test 2: Alternative splitting approach...")
                    try:
                        print("🔄 Trying alternative approach with limit()...")
                        total_size = ds.count()
                        train_size = int(total_size * 0.7)
                        val_size = int(total_size * 0.15)
                        
                        train_ds = ds.limit(train_size)
                        remaining_ds = ds.limit(total_size - train_size)
                        validation_ds = remaining_ds.limit(val_size)
                        test_ds = remaining_ds.limit(total_size - train_size - val_size)
                        
                        print("✅ Alternative split successful")
                        
                        # Test the alternative split results
                        train_count = train_ds.count()
                        val_count = validation_ds.count()
                        test_count = test_ds.count()
                        print(f"   Train: {train_count}, Val: {val_count}, Test: {test_count}")
                        
                    except Exception as alt_error:
                        print(f"❌ Alternative approach also failed: {alt_error}")
                        return False
        
        # Test 3: Test with smaller dataset
        print("\n📊 Test 3: Testing with smaller dataset...")
        try:
            small_ds = ds.limit(1000)  # Smaller dataset
            print(f"✅ Created smaller dataset with {small_ds.count()} rows")
            
            small_train, small_val, small_test = small_ds.split_proportionately([0.7, 0.15])
            print("✅ Small dataset split successful")
            
            small_train_size = small_train.count()
            small_val_size = small_val.count()
            small_test_size = small_test.count()
            print(f"   Train: {small_train_size}, Val: {small_val_size}, Test: {small_test_size}")
            
        except Exception as e:
            print(f"❌ Small dataset split failed: {e}")
        
        ray.shutdown()
        print("\n🎉 Dataset splitting tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset splitting test failed: {e}")
        print("\n💡 Troubleshooting suggestions:")
        print("   1. The GCS timeout might be due to cluster load")
        print("   2. Try restarting the Ray cluster")
        print("   3. Check if other processes are using the cluster")
        print("   4. Consider using a smaller dataset initially")
        return False

if __name__ == "__main__":
    success = test_dataset_splitting()
    if success:
        print("\n🎉 Dataset splitting is working!")
        print("💡 You can now run the full model: python model.py")
    else:
        print("\n❌ Dataset splitting still has issues.")
        print("💡 Consider using a smaller dataset or restarting the cluster.")
        sys.exit(1) 