---
title: Single Layer Perceptron
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

# Single Layer Perceptron

A single-layer neural network built without external ML libraries — only NumPy and OpenCV.

Use the Gradio app to capture webcam images for 4 classes, train the perceptron, classify live frames, and visualize learned weights.

## See-through mode

Trained weights are converted into color or black-and-white images so you can visually inspect what the network learned.

![sample image](s.webp)

## Windows version

For local use on Windows with keyboard controls, run `main_windows.py`:

| Key | Action |
|-----|--------|
| `q` | Gather training images (a/s/d/f for classes 0–3) |
| `z` | Train |
| `m` | Test / classify |
| `p` | Export weights |
| `esc` | Exit |

Requires the `keyboard` package and a local webcam.

## Help notes

- A neuron holds a number between 0 and 1 (its activation).
- Weights connect input neurons to output neurons; they can be positive or negative.
- Bias is a threshold for each output neuron.
- If weighted sum > bias, the neuron fires.

## Formula

```
(1×n) · (n×k) = (1×k)
```

Where `n` is the number of input neurons and `k` is the number of output neurons.

**ReLU:** `max(0, x)`  
**Sigmoid:** `1 / (1 + e^-x)`
