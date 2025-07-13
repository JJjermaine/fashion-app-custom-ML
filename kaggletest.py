import kagglehub

# Download latest version
path = kagglehub.dataset_download("thusharanair/deepfashion2-original-with-dataframes")

print("Path to dataset files:", path)