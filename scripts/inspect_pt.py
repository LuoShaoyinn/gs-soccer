import torch
import os, sys

def print_model_dimensions(model):
    print(f"{'Layer Name':<25} | {'Weight Shape':<20} | {'Bias Shape':<15}")
    print("-" * 65)
    
    # Iterate through all named parameters (weights and biases)
    for name, param in model.named_parameters():
        # param.shape will give you [out_features, in_features] for Linear weights
        shape_str = str(list(param.shape))
        print(f"{name:<25} | {shape_str:<20}")

def inspect_pt_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"--- Inspecting: {file_path} ---")
    
    try:
        # map_location='cpu' ensures it loads even if saved on a GPU you don't have
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        
        print(f"Object Type: {type(data)}")
        print("-" * 30)

        if isinstance(data, dict):
            print(f"Dictionary detected with {len(data)} keys.")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: [ {key:20} ] | Type: Tensor | Shape: {list(value.shape)} | Dtype: {value.dtype}")
                elif isinstance(value, (dict, list)):
                    print(f"Key: [ {key:20} ] | Type: {type(value).__name__} | Length: {len(value)}")
                else:
                    print(f"Key: [ {key:20} ] | Type: {type(value).__name__} | Value: {value}")

        elif isinstance(data, torch.Tensor):
            print("Direct Tensor detected.")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Sample data:\n{data[:2]}") # Show first two rows/elements

        elif hasattr(data, 'state_dict'):
            print("Full PyTorch Model/Object detected.")
            print("Architecture Summary:")
            print(data)
            
        else:
            print("Unknown/Custom Object structure.")
            print("Attributes found:", dir(data))

        print_model_dimensions(data)

    except Exception as e:
        print(f"Failed to load .pt file: {e}")

if __name__ == "__main__":
    # Replace with your actual file path
    path = sys.argv[1] 
    inspect_pt_file(path)
