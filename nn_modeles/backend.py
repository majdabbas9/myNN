import sys
import warnings

# Suppress CuPy warnings if possible, or let them show so user knows whats happening
try:
    import cupy as np
    # Try a simple operation to ensure CUDA is actually working (catch DLL errors)
    np.cuda.runtime.getDeviceCount()
    np.array([1]).device
    dummy = np.random.randn(1)
    
    print(f"Using CuPy (GPU) - {np.cuda.runtime.getDeviceCount()} device(s) found.")
except (ImportError, Exception) as e:
    # If CuPy is missing or CUDA DLLs are missing (ImportError or other exceptions during usage)
    print(f"CuPy import failed or CUDA execution failed: {e}")
    print("Falling back to NumPy (CPU).")
    import numpy as np

def to_numpy(array):
    """Converts a backend array to a NumPy array."""
    if 'cupy' in sys.modules and hasattr(array, 'get'): # Check for .get() method typical of cupy arrays
        return array.get()
    
    # Fallback if somehow it's a cupy array but check failed, or just numpy
    # np.asnumpy handles cupy arrays if np is cupy.
    # If np is numpy, it doesn't have asnumpy. 
    
    if 'cupy' in sys.modules and sys.modules['cupy'] == np:
         return np.asnumpy(array)
         
    return np.array(array)

def to_tensor(array):
    """Converts a NumPy array to a backend array."""
    return np.array(array)
