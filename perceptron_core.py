import cv2
import numpy as np


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


ACTIVATIONS = {"relu": relu, "sigmoid": sigmoid}


def channels(color: bool) -> int:
    return 3 if color else 1


def init_weights(resize: int, color: bool, n_classes: int) -> np.ndarray:
    return np.zeros((resize ** 2 * channels(color), n_classes))


def image_to_neurons(image: np.ndarray, resize: int, color: bool) -> np.ndarray:
    ch = channels(color)
    if color:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
    else:
        image = image if image.ndim == 2 else np.mean(image, axis=2).astype(np.uint8)
    flat = np.ravel(cv2.resize(image, (resize, resize))).astype(np.float64)
    neurons = np.zeros(resize ** 2 * ch)
    neurons[: len(flat)] = flat
    return neurons / 255


def predict_scores(image, weights, activation, resize, color) -> list[float]:
    output = np.dot(image_to_neurons(image, resize, color), weights)
    return [float(activation(x)) for x in output]


def train_weights(samples, weights, bias, activation, resize, color):
    shuffled = samples.copy()
    np.random.shuffle(shuffled)
    for label, image in shuffled:
        inputs = image_to_neurons(image, resize, color)
        output = [activation(x) for x in np.dot(inputs, weights)]
        if output[label] < bias:
            weights.T[label] += inputs
        for r, score in enumerate(output):
            if score > bias and r != label:
                weights.T[r] -= inputs
    return weights


def scores_to_percentages(raw_scores, activation_name) -> list[float]:
    if activation_name == "sigmoid":
        return [round(min(max(s, 0.0), 1.0) * 100, 1) for s in raw_scores]
    peak = max(raw_scores) if raw_scores else 0.0
    if peak <= 0:
        return [0.0] * len(raw_scores)
    return [round(s / peak * 100, 1) for s in raw_scores]


def weights_to_images(weights, resize, color):
    ch = channels(color)
    images = []
    for i in range(weights.shape[1]):
        flat = np.clip(weights.T[i] * 255, 0, 255).astype(np.uint8)
        shape = (resize, resize, 3) if ch == 3 else (resize, resize)
        images.append(flat.reshape(shape))
    return images
