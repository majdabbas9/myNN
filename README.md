# Neural Network from Scratch

A custom implementation of a deep learning framework built from scratch in Python, capable of training neural networks on datasets like MNIST and Spiral Data.

## Features

- **Custom Layers**: `LayerDense` implementation for fully connected layers.
- **Activation Functions**: ReLU, Softmax.
- **Optimizers**:
    - SGD (Stochastic Gradient Descent)
    - SGD with Momentum
    - AdaGrad
    - Adam
- **Loss Functions**: Categorical Crossentropy (with integrated Softmax activation for stability).
- **Backend Support**: Abstraction layer (`backend.py`) to potentially support different array backends (e.g., NumPy).
- **Examples**:
    - `main.py`: Classification on generated Spiral Data.
    - `test_minist.py`: Handwritten digit recognition on the MNIST dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/majdabbas9/myNN.git
   cd myNN
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn nnfs
   ```

## Usage

### Train on Spiral Data
Run various experiments on synthetic spiral data to test the network's ability to learn non-linear boundaries.

```bash
python main.py
```

### Train on MNIST (Handwritten Digits)
Train a multi-layer perceptron to recognize handwritten digits. This script fetches the MNIST dataset, creates a deep network (784 -> 128 -> 64 -> 10), and uses mini-batch gradient descent.

```bash
python test_minist.py
```

## Structure

- `nn_Layer.py`: Contains the dense layer implementation (`LayerDense`).
- `optimizers.py`: Contains optimizer implementations.
- `activation_fucntions.py`: Activation functions like ReLU.
- `loss_soft_max_activation.py`: Combined Softmax activation and Categorical Crossentropy loss for efficient backpropagation.
- `backend.py`: Helper functions for tensor/array management.
- `accuracy.py`: Utility to calculate model accuracy.

---
Created by [Majd Abbas](https://github.com/majdabbas9).
