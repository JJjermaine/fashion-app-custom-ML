# Set environment variables for Ray timeouts BEFORE any imports
import os
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "60"  # Increase from default 5s to 60s
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "60"    # Increase connection timeout
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "120"   # Increase request timeout
os.environ["RAY_gcs_rpc_server_retry_timeout_s"] = "20"      # Increase retry timeout
os.environ["RAY_gcs_rpc_server_retry_interval_s"] = "2"      # Retry interval
os.environ["RAY_gcs_rpc_server_max_retries"] = "10"          # Maximum retries

# Ray Data specific timeouts
os.environ["RAY_DATA_READ_TIMEOUT_S"] = "120"                # Dataset read timeout
os.environ["RAY_DATA_WRITE_TIMEOUT_S"] = "120"               # Dataset write timeout

# Additional Ray timeouts
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "120"   # Longer timeout for dataset ops
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "180"   # Longer request timeout
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "120" # Longer reconnect timeout

# Ray cluster timeouts
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "180"   # Very long timeout for cluster ops
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "300"   # Very long request timeout

import ray
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from google.cloud import aiplatform
import time

# Correct imports for modern Ray AIR
from ray.data.preprocessors import TorchVisionPreprocessor
from ray.air import session
from ray.train import Checkpoint
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer, TorchPredictor
from sklearn.metrics import classification_report

# It's good practice to handle potential Kaggle Hub import errors
try:
    import kagglehub
except ImportError:
    print("Kaggle Hub not installed. Please run: pip install kagglehub")
    kagglehub = None

def setup_ray_cluster():
    """Initializes a connection to the Ray Cluster with improved error handling and timeouts."""
    
    print("🔧 Setting up Ray cluster connection with improved timeouts...")
    
    try:
        # Initialize Vertex AI
        aiplatform.init(project='fashion-app-f2861', location='us-west2')
        print("✅ Vertex AI initialized successfully")
        
        # Try to connect to the Ray cluster with retry logic
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries} to connect to Ray cluster...")
                
                # Use the Vertex AI Ray address
                ray_address = "vertex_ray://projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801"
                
                # Initialize Ray with only supported parameters for Vertex AI
                ray.init(
                    address=ray_address,
                    log_to_driver=True,
                    ignore_reinit_error=True
                )
                
                print("✅ Connected to Ray cluster successfully!")
                
                # Verify the connection by checking cluster status
                try:
                    cluster_resources = ray.cluster_resources()
                    print(f"📊 Cluster resources: {cluster_resources}")
                    return True
                except Exception as status_error:
                    print(f"⚠️  Connected but couldn't get cluster status: {status_error}")
                    # Still return True since we're connected
                    return True
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("❌ All connection attempts failed.")
                    return False
                        
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI or Ray cluster: {e}")
        return False


def load_and_prep_data(base_path):
    """Loads the DeepFashion2 dataset using Ray Datasets with retry logic."""
    print("Loading and preparing DeepFashion2 dataset...")
    
    # The CSV file contains mappings from image filenames to categories.
    # Reverting to your original CSV path logic
    df_train = pd.read_csv(os.path.join(base_path, "img_info_dataframes", "train.csv"))

    # Map category names to integer labels for the model.
    df_train['category_id'] = pd.Categorical(df_train['category_name']).codes
    class_names = list(pd.Categorical(df_train['category_name']).categories)
    num_classes = len(class_names)

    print(f"Found {num_classes} classes.")

    # Convert Kaggle paths to local paths
    def convert_kaggle_path_to_local(kaggle_path):
        # Extract filename from Kaggle path like: /kaggle/input/deepfashion2-original-with-dataframes/DeepFashion2/deepfashion2_original_images/train/image/000001.jpg
        filename = os.path.basename(kaggle_path)
        return os.path.join(base_path, 'deepfashion2_original_images', 'train', 'image', filename)

    df_train['image'] = df_train['path'].apply(convert_kaggle_path_to_local)
    
    # Filter out rows where the image file does not exist
    df_train = df_train[df_train['image'].apply(os.path.exists)]

    # Create a Ray Dataset from the pandas DataFrame with retry logic
    print("Creating Ray dataset from pandas DataFrame...")
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            ds = ray.data.from_pandas(df_train[['image', 'category_id']])
            print("✅ Ray dataset created successfully")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to create dataset failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Failed to create Ray dataset after all attempts")
                return None, None, None, num_classes, class_names
    
    # Split the data into training, validation, and a final test set with retry logic
    print("Splitting dataset into train/validation/test sets...")
    max_retries = 5  # Increased from 3 to 5
    retry_delay = 10  # Increased from 5 to 10
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries} to split dataset...")
            train_ds, validation_ds, test_ds = ds.split_proportionately([0.7, 0.15])
            print("✅ Dataset split successful")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to split dataset failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                print(f"⏳ Next retry delay will be {retry_delay} seconds")
            else:
                print("❌ Failed to split dataset after all attempts")
                print("💡 This might be due to GCS timeout issues. Trying alternative approach...")
                
                # Try alternative splitting approach
                try:
                    print("🔄 Trying alternative dataset splitting approach...")
                    # Use a simpler split method
                    total_size = ds.count()
                    train_size = int(total_size * 0.7)
                    val_size = int(total_size * 0.15)
                    
                    train_ds = ds.limit(train_size)
                    remaining_ds = ds.limit(total_size - train_size)
                    validation_ds = remaining_ds.limit(val_size)
                    test_ds = remaining_ds.limit(total_size - train_size - val_size)
                    
                    print("✅ Alternative dataset split successful")
                except Exception as alt_error:
                    print(f"❌ Alternative splitting also failed: {alt_error}")
                    return None, None, None, num_classes, class_names
    
    # Get dataset sizes with retry logic
    print("Getting dataset sizes...")
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            train_size = train_ds.count()
            val_size = validation_ds.count()
            test_size = test_ds.count()
            
            print(f"Train dataset size: {train_size}")
            print(f"Validation dataset size: {val_size}")
            print(f"Test dataset size: {test_size}")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to get dataset sizes failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("⚠️  Could not get dataset sizes, but continuing...")
                break
    
    return train_ds, validation_ds, test_ds, num_classes, class_names


def get_pytorch_model(num_classes):
    """Defines the PyTorch model using a pre-trained ResNet18."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_loop_per_worker(config):
    """The core training function that Ray Train will execute on each worker."""
    lr = config["lr"]
    epochs = config["epochs"]
    num_classes = config["num_classes"]
    
    train_data_shard = session.get_dataset_shard("train")
    
    model = get_pytorch_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    
    model = ray.train.torch.prepare_model(model)
    
    for epoch in range(epochs):
        model.train()
        # The batch now correctly contains image tensors, not file paths.
        for batch in train_data_shard.iter_torch_batches(batch_size=32, dtypes=torch.float32):
            inputs, labels = batch["image"], batch["category_id"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        val_data_shard = session.get_dataset_shard("validation")
        with torch.no_grad():
            for batch in val_data_shard.iter_torch_batches(batch_size=32, dtypes=torch.float32):
                inputs, labels = batch["image"], batch["category_id"]
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total if total > 0 else 0
        
        checkpoint = Checkpoint.from_model(model)
        session.report({"val_accuracy": val_accuracy}, checkpoint=checkpoint)

# *** MEMORY-EFFICIENT IMAGE LOADING FUNCTION ***
def load_images_from_paths(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Ray Data UDF: Takes a batch of file paths and returns a batch of image data.
    Memory-efficient version that processes smaller batches.
    """
    try:
        # Process images one by one to avoid memory issues
        images = []
        for path in batch["image"]:
            try:
                img = Image.open(path).convert("RGB")
                # Resize immediately to reduce memory usage
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.uint8)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create a placeholder image if loading fails
                placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
                images.append(placeholder)
        
        batch["image"] = images
        return batch
    except Exception as e:
        print(f"Error in load_images_from_paths: {e}")
        # Return batch with placeholder images if processing fails
        batch["image"] = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(len(batch))]
        return batch

def main():
    """Main function to run the fashion ML pipeline."""
    print("Fashion Outfit Classification with Ray AIR Starting...")
    print("=" * 60)
    
    if setup_ray_cluster() == False:
        exit(1)
    
    # Reverting to your original local path logic
    print("Loading dataset from local path...")
    dataset_path = r"C:\Users\JJjer\.cache\kagglehub\datasets\thusharanair\deepfashion2-original-with-dataframes\versions\2\DeepFashion2"
    print(f"Dataset loaded from: {dataset_path}")
    
    train_ds, val_ds, test_ds, num_classes, class_names = load_and_prep_data(dataset_path)

    if train_ds is None:
        print("❌ Halting execution due to data loading failure.")
        return

    # *** MEMORY-EFFICIENT DATA PROCESSING ***
    # Use smaller batches and process data more efficiently
    print("Converting file paths to image data (memory-efficient)...")
    
    # Reduce dataset size for testing - you can increase this later
    print("Using smaller dataset for memory efficiency...")
    
    # Apply dataset limits with retry logic
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            train_ds = train_ds.limit(1000)  # Start with 1000 images
            val_ds = val_ds.limit(200)       # Start with 200 images  
            test_ds = test_ds.limit(200)     # Start with 200 images
            print("✅ Dataset limits applied successfully")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to apply dataset limits failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Failed to apply dataset limits after all attempts")
                return
    
    # Process with smaller batch sizes with retry logic
    print("Processing images with map_batches...")
    for attempt in range(max_retries):
        try:
            train_ds = train_ds.map_batches(load_images_from_paths, batch_format="pandas", batch_size=10)
            val_ds = val_ds.map_batches(load_images_from_paths, batch_format="pandas", batch_size=10)
            test_ds = test_ds.map_batches(load_images_from_paths, batch_format="pandas", batch_size=10)
            print("✅ Image processing completed successfully")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to process images failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Failed to process images after all attempts")
                return

    # Define the preprocessor that will be applied to the datasets.
    # Since we're already resizing images in load_images_from_paths, we only need normalization
    preprocessor = TorchVisionPreprocessor(
        columns=["image"], # This column now contains actual image data.
        transform=transforms.Compose([
            transforms.ToTensor(), # Convert NumPy array to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # Apply the preprocessor to the datasets with retry logic
    print("Applying image preprocessing...")
    for attempt in range(max_retries):
        try:
            train_ds = preprocessor.fit_transform(train_ds)
            val_ds = preprocessor.transform(val_ds)
            test_ds = preprocessor.transform(test_ds)
            print("✅ Image preprocessing completed successfully")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} to apply preprocessing failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("❌ Failed to apply preprocessing after all attempts")
                return

    print("\nTraining model with Ray Train...")
    
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=torch.cuda.is_available(),
    )

    # Create the trainer without the preprocessor parameter
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"lr": 0.001, "epochs": 5, "num_classes": num_classes},
        scaling_config=scaling_config,
        datasets={"train": train_ds, "validation": val_ds},
    )

    result = trainer.fit()

    print(f"\nTraining complete. Best validation accuracy: {result.metrics.get('val_accuracy', 0):.4f}")
    best_checkpoint = result.checkpoint
    if best_checkpoint:
        print(f"   Best model checkpoint stored at: {best_checkpoint.path}")

        print("\n🔬 Running final evaluation on the unseen test set...")

        # Create predictor and make predictions on the preprocessed test dataset
        predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        
        predictions_df = predictor.predict(test_ds)

        true_labels = predictions_df.select("category_id").to_pandas()["category_id"]
        predicted_labels = predictions_df.select("predictions").to_pandas()["predictions"].apply(lambda x: x.argmax())

        print("\n" + "=" * 60)
        print("🎉 Fashion ML Pipeline Complete! 🎉")
        print("\nClassification Report on Test Data:")
        
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=class_names,
            zero_division=0
        )
        print(report)
    else:
        print("❌ Training did not produce a valid checkpoint.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
