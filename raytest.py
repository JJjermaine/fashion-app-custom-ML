import os
import sys
import ray
from google.cloud import aiplatform

# Version compatibility settings

print("🔍 Testing remote Ray initialization...")
print(f"Local Ray version: {ray.__version__}")

# Initialize Vertex AI
aiplatform.init(project='fashion-app-f2861', location='us-west2')

# Define the address for the remote Ray cluster
# Try different connection formats
address = 'vertex_ray://projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801'

# Alternative: Try using the standard Ray client format
# address = 'ray://cluster-20250712-190801.us-west2.google.com:10001'



try:
    # Attempt to connect to the remote Ray cluster
    print(f"☁️ Attempting to connect to Ray cluster at: {address}")
    
    # Try with specific configuration to avoid version issues
    ray.init(
        address=address
    )
    
    # If successful, print confirmation and resource details
    print("✅ Remote Ray connected successfully!")
    print(f"Cluster resources: {ray.cluster_resources()}")
    print(f"Available nodes: {ray.nodes()}")

except Exception as e:
    # If the connection fails, print detailed error messages and exit
    print(f"❌ Failed to connect to remote Ray cluster: {e}")
    print("\n💡 Troubleshooting tips:")
    print("1. Check Ray version compatibility: pip install ray==2.9.0")
    print("2. Verify cluster is running: gcloud ai persistent-resources list --region=us-west2")
    print("3. Check authentication: gcloud auth list")
    print("4. Try updating google-cloud-aiplatform: pip install --upgrade google-cloud-aiplatform")
    
    # Exit the script with a non-zero status code to indicate failure
    sys.exit(1)

# This part will only be reached if the initialization was successful
if ray.is_initialized():
    print("\n🎉 Ray is initialized and ready to use!")