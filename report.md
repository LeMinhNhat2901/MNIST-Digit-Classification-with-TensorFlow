# MNIST Classification Report

## Results Summary

Table copied from `results/summary.md`:

| Model | Weights | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| logistic | 7,850 | 0.8995 | 0.8993 | 0.8995 | 0.8991 |
| nn | 109,386 | 0.9781 | 0.9782 | 0.9781 | 0.9781 |

## Model Comparison

### Accuracy Performance
The **Neural Network (nn)** achieved a higher accuracy compared to the **Logistic Regression (logistic)** model. 
The Neural Network reached **97.81%** accuracy, which is exactly **7.86%** higher than the Logistic model's **89.95%**. 

### Weight (Parameter) Comparison
- **Logistic Regression**: 7,850 weights
- **Neural Network**: 109,386 weights

The Neural Network contains nearly **14 times more weights** than the Logistic model. The Logistic model is exceptionally lightweight because it maps the 784 input pixels directly to the 10 output classes linearly without intermediate representations (784 * 10 weight coefficients + 10 biases = 7,850 parameters). In contrast, the Neural Network features two hidden dense layers (128 units and 64 units) which add substantially more parameters and computational density.

### Trade-off Between Complexity and Accuracy
The results highlight a fundamental machine learning trade-off: **capacity vs. cost**.
Higher complexity (more weights, more layers) allows the network to learn rich, non-linear, hierarchical patterns, resulting in superior generalization and accuracy (the huge bump from ~90% to ~98%). However, this comes at the cost of:
1. **Computational Overhead**: Slower training and inference times.
2. **Memory/Storage**: A larger memory footprint is required to serve the model.
3. **Overfitting**: Although not a major issue with the large MNIST dataset, highly parameterized models are generally natively more prone to memorizing the training set (high variance) without sufficient regularization. 

## Class-wise Performance (Struggling Digits)

By inspecting the per-class classification reports and confusion matrices, we can pinpoint specific weaknesses:

- **Logistic Model**: Struggles most with **Digit 5** (Lowest F1-score: `0.8487`, and a notably low recall of `0.8240`). Consulting the confusion matrix shows that true 5s are frequently misclassified as **3** (41 occurrences) and **8** (38 occurrences). It also struggles with **Digit 8** (F1-score: `0.8524`).
- **Neural Network Model**: Although highly accurate across the board, it performs relatively worst on **Digit 9** (Lowest F1-score: `0.9694`) and **Digit 3** (F1-score: `0.9743`). For instance, true 7s are occasionally misclassified as 1s or 2s, and some overlapping confusion exists between 4, 7, and 9 due to similar handwritten structural traits.

## Suggested Improvement

**Recommended Improvement:** Introduce **Convolutional Neural Network (CNN)** layers (e.g., Keras `Conv2D` and `MaxPooling2D`) before the Dense classification layers.

**Why it helps:** Both current models take the image and immediately flatten it into a 1-dimensional array, effectively destroying the natural 2D spatial relationships and locality between neighboring pixels. CNNs use 2D localized filters that slide across the image, allowing the network to explicitly capture and preserve structural hierarchies—like edges, curves, loops, and corners. This architectural change injects a strong visual prior that makes image recognition vastly more effective, frequently pushing MNIST accuracies above 99.0%.
