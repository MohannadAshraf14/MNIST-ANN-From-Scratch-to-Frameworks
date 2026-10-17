# MNIST Handwritten Digit Classification — ANN in NumPy, TensorFlow & PyTorch

An implementation of the **same fully connected Artificial Neural Network (ANN)** using three approaches: **NumPy (from scratch)**, **TensorFlow/Keras**, and **PyTorch**. The goal is to understand how neural networks work internally and how deep learning frameworks simplify training.

## Project overview

The model classifies handwritten digits (0–9) from the MNIST dataset. Each 28 × 28 grayscale image is flattened into 784 input features and normalized to the range [0, 1].

**Architecture:** `784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax probabilities / logits)`

- Input: 784 pixel values
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU
- Output: 10 classes
- Loss: categorical cross-entropy (NumPy), sparse categorical cross-entropy (TensorFlow), or cross-entropy on logits (PyTorch)

In the PyTorch implementation, the model returns **logits** rather than applying Softmax, because `nn.CrossEntropyLoss` handles the required normalization internally.

## Implementations

| Implementation | Training approach | Gradient computation | Optimizer |
|---|---|---|---|
| NumPy | Manual mini-batch training loop | Manual backpropagation, verified with numerical gradient checking | Mini-batch SGD |
| TensorFlow/Keras | `model.fit()` | Automatic differentiation | Adam |
| PyTorch | Explicit training loop using `DataLoader` | Autograd via `loss.backward()` | Adam |

## Results

| Implementation | Learning rate | Test accuracy | Test loss |
|---|---:|---:|---:|
| NumPy | 0.01 | **96.44%** | Not recorded |
| TensorFlow/Keras (Adam) | 0.01 | 96.24% | 0.1348 |
| TensorFlow/Keras (Adam) | 0.001 | **97.73%** | 0.0758 |
| PyTorch (Adam) | 0.001 | **97.72%** | 0.0825 |

**Important:** These are results from individual runs, not averages across multiple seeds. The NumPy run used SGD, whereas the TensorFlow and PyTorch runs used Adam; random initialization and data ordering may also differ. The table illustrates outcomes from these implementations rather than a controlled framework benchmark.

### Training observations

- The NumPy implementation was checked using **numerical gradient checking** to validate backpropagation.
- Lowering TensorFlow/Adam's learning rate from `0.01` to `0.001` improved the observed test accuracy from **96.24%** to **97.73%** in these runs.
- The PyTorch validation loss reached its lowest observed value (**0.0805**) at epoch 8. Although training loss continued to decrease, validation loss subsequently rose, suggesting overfitting.
- **Early stopping** in PyTorch stopped training at epoch 12 and restored the weights from epoch 8, based on validation loss.

## Dataset

[MNIST in CSV format on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

- 60,000 labeled training images, split into **54,000 training** and **6,000 validation** samples
- 10,000 test images
- Each image: 28 × 28 pixels, flattened to 784 features
- Labels: integer digits from 0 to 9

The TensorFlow and PyTorch notebooks use a NumPy random generator with seed `42` for the train/validation split. To reproduce the same split in NumPy, use the identical indices. The original NumPy experiment may have used a different split.

## Repository structure

```text
MNIST-ANN-From-Scratch-to-Frameworks/
├── notebooks/
│   ├── ANN_NumPy.ipynb
│   ├── ANN_TensorFlow.ipynb
│   └── ANN_PyTorch.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the notebooks

1. Download or add the Kaggle dataset linked above.
2. Open the notebook for the implementation you want to run (Kaggle or Jupyter).
3. Update the CSV paths to match your environment. The original Kaggle path used in these notebooks is:
   ```text
   /kaggle/input/datasets/oddrationale/mnist-in-csv/
   ```
4. Run the notebook cells in order.

For local use, install the dependencies in `requirements.txt` and ensure the MNIST CSV files are available. GPU acceleration is optional for this small ANN.

## Key takeaways

Implementing an ANN with NumPy provides direct experience with matrix operations, activation functions, loss computation, backpropagation, and parameter updates. TensorFlow/Keras makes model training concise, while PyTorch retains a more explicit training loop with automatic gradient computation. All three approaches can learn the MNIST classification task effectively.

## Future improvements

- Compare optimizers using identical data splits, seeds, and training settings
- Add confusion matrices and examples of misclassified digits
- Test regularization techniques such as dropout or weight decay
- Compare with a convolutional neural network (CNN)
