import time
import numpy as np
import math
from scipy.spatial.distance import pdist, cdist

def old_distance(r1, r2):
    delta_v = np.array(r1) - np.array(r2)
    dist = np.linalg.norm(delta_v)
    return dist

def new_distance(r1, r2):
    return math.dist(r1, r2)

def benchmark():
    N = 500  # Number of atoms
    # Generate random coordinates
    coords = np.random.rand(N, 3).tolist()
    
    print(f"Benchmarking with N={N} atoms ({N*(N-1)//2} pairs)...")
    
    # 1. Old distance function (nested loop)
    start_time = time.time()
    old_distances = []
    for i in range(N):
        for j in range(i+1, N):
            old_distances.append(old_distance(coords[i], coords[j]))
    old_time = time.time() - start_time
    print(f"1. Old nested loop with np.array(): {old_time:.4f} seconds")
    
    # 2. New distance function with math.dist (nested loop)
    start_time = time.time()
    new_distances = []
    for i in range(N):
        for j in range(i+1, N):
            new_distances.append(new_distance(coords[i], coords[j]))
    new_time = time.time() - start_time
    print(f"2. New nested loop with math.dist:  {new_time:.4f} seconds ({old_time/new_time:.1f}x speedup)")
    
    # 3. Fully vectorized with scipy pdist
    start_time = time.time()
    coords_array = np.array(coords) # Include array conversion overhead
    vec_distances = pdist(coords_array)
    vec_time = time.time() - start_time
    print(f"3. Vectorized with scipy.pdist:     {vec_time:.4f} seconds ({old_time/vec_time:.1f}x speedup)")

if __name__ == "__main__":
    benchmark()
