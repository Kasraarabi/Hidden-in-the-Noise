"""
This code evaluates the empirical computational cost associated with
searching through a vast number of noise vectors.
Specifically, it generates 100,000 random noise vectors and measures 
the time required to identify the closest match to a newly generated random noise vector.
"""

import torch
import time

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the parameters
num_noises = 100000
noise_shape = (4, 64, 64)

# Generate 100,000 Gaussian random noises
noises = torch.randn(num_noises, *noise_shape, device=device)

# Generate a new noise
new_noise = torch.randn(1, *noise_shape, device=device)

# Function to compute L2 distances and find the minimum
def compute_min_distance(noises, new_noise):
    # Flatten the noises for easier computation
    flat_noises = noises.view(num_noises, -1)
    flat_new_noise = new_noise.view(1, -1)
    
    # Compute L2 distances
    distances = torch.norm(flat_noises - flat_new_noise, dim=1)
    
    # Find the minimum distance
    min_distance = torch.min(distances)
    
    return min_distance

# Warm-up run (to ensure GPU is initialized)
_ = compute_min_distance(noises, new_noise)

# Measure execution time
start_time = time.time()

min_distance = compute_min_distance(noises, new_noise)

end_time = time.time()
execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

print(f"Execution time: {execution_time:.2f} ms")

# Print memory usage
print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")