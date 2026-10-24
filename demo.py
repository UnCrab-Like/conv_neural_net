import base64
import gzip
import io
from collections.abc import Callable

import cupy as cp
from PIL import Image

from src.back_prop import back_propagation
from src.back_prop_old import back_propagation as back_propagation_v1
from src.front_prop import front_propagation
from src.front_prop_old import front_propagation as front_propagation_v1


def _load_training_data(sample_count: int) -> tuple[cp.ndarray, cp.ndarray]:
    with gzip.open("training_data/train-images-idx3-ubyte.gz", "rb") as images_file:
        images_file.read(16)
        images = cp.frombuffer(images_file.read(), dtype=cp.uint8)
    with gzip.open("training_data/train-labels-idx1-ubyte.gz", "rb") as labels_file:
        labels_file.read(8)
        labels = cp.frombuffer(labels_file.read(), dtype=cp.uint8)

    images = images.astype(cp.float32).reshape(-1, 28, 28) / 255.0
    order = cp.random.permutation(len(images))[:sample_count]
    return images[order], labels[order].astype(cp.int32)


def _load_validation_data(sample_count: int) -> tuple[cp.ndarray, cp.ndarray]:
    with gzip.open("t10k-images-idx3-ubyte.gz", "rb") as images_file:
        images_file.read(16)
        images = cp.frombuffer(images_file.read(), dtype=cp.uint8)
    with gzip.open("t10k-labels-idx1-ubyte.gz", "rb") as labels_file:
        labels_file.read(8)
        labels = cp.frombuffer(labels_file.read(), dtype=cp.uint8)

    images = images.astype(cp.float32).reshape(-1, 28, 28) / 255.0
    order = cp.random.permutation(len(images))[:sample_count]
    return images[order], labels[order].astype(cp.int32)


def _kaiming_uniform(shape: tuple[int, ...], fan_in: int) -> cp.ndarray:
    limit = cp.sqrt(6.0 / fan_in)
    return cp.random.uniform(-limit, limit, size=shape).astype(cp.float32)


def _image_data_url(values: cp.ndarray) -> str:
    image_values = cp.asnumpy(values).astype("float32")
    minimum = float(image_values.min())
    maximum = float(image_values.max())
    if maximum > minimum:
        image_values = (image_values - minimum) * (255.0 / (maximum - minimum))
    else:
        image_values = image_values * 0

    image = Image.fromarray(image_values.clip(0, 255).astype("uint8"), mode="L")
    image = image.resize((196, 196), Image.Resampling.NEAREST)
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _predict_probabilities(
    model: str,
    images: cp.ndarray,
    conv_filter: cp.ndarray,
    dense_layer_1: cp.ndarray,
    bias_hidden: cp.ndarray,
    dense_layer_2: cp.ndarray,
    bias_out: cp.ndarray,
) -> cp.ndarray:
    front = front_propagation if model == "v2" else front_propagation_v1
    batch_conv = front.filter_image(images, conv_filter, "L")
    if model == "v1":
        batch_conv = batch_conv[:, None, :, :]
    activated = front.leaky_RELU(batch_conv, 0.001)
    if model == "v1":
        pooled = front.max_pooling(activated[:, 0], "L")[:, None]
    else:
        pooled = front.max_pooling(activated, "L")
    flattened = front.leaky_RELU(pooled.reshape(len(images), -1), 0.01)
    hidden = front.leaky_RELU(
        front.forward_pass(flattened, dense_layer_1, bias_hidden), 0.001
    )
    logits = front.forward_pass(hidden, dense_layer_2, bias_out)
    probabilities = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
    return probabilities / cp.sum(probabilities, axis=1, keepdims=True)


def _create_preview(
    model: str,
    image: cp.ndarray,
    label: int,
    conv_filter: cp.ndarray,
    dense_layer_1: cp.ndarray,
    bias_hidden: cp.ndarray,
    dense_layer_2: cp.ndarray,
    bias_out: cp.ndarray,
) -> dict[str, object]:
    front = front_propagation if model == "v2" else front_propagation_v1
    batch_conv = front.filter_image(image[None, :, :], conv_filter, "L")
    if model == "v1":
        batch_conv = batch_conv[:, None, :, :]
    activated = front.leaky_RELU(batch_conv, 0.001)
    if model == "v1":
        pooled = front.max_pooling(activated[:, 0], "L")[:, None]
    else:
        pooled = front.max_pooling(activated, "L")
    flattened = front.leaky_RELU(pooled.reshape(1, -1), 0.01)
    hidden = front.leaky_RELU(
        front.forward_pass(flattened, dense_layer_1, bias_hidden), 0.001
    )
    logits = front.forward_pass(hidden, dense_layer_2, bias_out)
    probabilities = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
    probabilities /= cp.sum(probabilities, axis=1, keepdims=True)
    prediction = int(cp.argmax(probabilities[0]).get())

    return {
        "label": label,
        "prediction": prediction,
        "confidence": float(cp.max(probabilities[0]).get()),
        "images": {
            "Input image": _image_data_url(image),
            "Convolution (filter 1)": _image_data_url(batch_conv[0, 0]),
            "Leaky ReLU": _image_data_url(activated[0, 0]),
            "Max pooling": _image_data_url(pooled[0, 0]),
        },
    }


def train_model(
    model: str,
    epochs: int,
    batch_size: int,
    sample_count: int,
    validation_count: int,
    learning_rate: float,
    on_epoch: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    if model not in {"v1", "v2"}:
        raise ValueError("Model must be 'v1' or 'v2'.")

    filter_count = 1 if model == "v1" else 16
    front = front_propagation if model == "v2" else front_propagation_v1
    back = back_propagation if model == "v2" else back_propagation_v1
    training_data, training_labels = _load_training_data(sample_count)
    validation_data, validation_labels = _load_validation_data(validation_count)

    conv_filter = _kaiming_uniform((filter_count, 3, 3), 9)
    dense_layer_1 = _kaiming_uniform((filter_count * 169, 128), filter_count * 169)
    dense_layer_2 = _kaiming_uniform((128, 10), 128)
    bias_hidden = cp.zeros((1, 128), dtype=cp.float32)
    bias_out = cp.zeros((1, 10), dtype=cp.float32)

    optimizer = None
    if model == "v2":
        from src.adam_optimizer import AdamOptimizer

        optimizer = AdamOptimizer(lr=learning_rate)

    history = []
    validation_history = []
    validation_accuracy = []
    for epoch in range(epochs):
        order = cp.random.permutation(sample_count)
        total_loss = 0.0
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            batch_images = training_data[indices]
            batch_labels = training_labels[indices]
            actual_batch_size = len(indices)

            batch_conv = front.filter_image(batch_images, conv_filter, "L")
            if model == "v1":
                batch_conv = batch_conv[:, None, :, :]
            batch_activated = front.leaky_RELU(batch_conv, 0.001)
            if model == "v1":
                batch_pooled = front.max_pooling(batch_activated[:, 0], "L")[:, None]
            else:
                batch_pooled = front.max_pooling(batch_activated, "L")
            batch_flat = batch_pooled.reshape(actual_batch_size, -1)
            batch_relu_2 = front.leaky_RELU(batch_flat, 0.01)
            hidden_pre_activation = front.forward_pass(
                batch_relu_2, dense_layer_1, bias_hidden
            )
            hidden_activations = front.leaky_RELU(hidden_pre_activation, 0.001)
            logits = front.forward_pass(hidden_activations, dense_layer_2, bias_out)

            probabilities = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
            probabilities /= cp.sum(probabilities, axis=1, keepdims=True)
            errors = probabilities - cp.eye(10, dtype=cp.float32)[batch_labels]

            grad_dense_2, grad_bias_out, _ = back.backpropagate(
                dense_layer_2, hidden_activations, errors, actual_batch_size
            )
            hidden_errors = (errors @ dense_layer_2.T) * back.deriv_leaky_RELU(
                hidden_pre_activation, 0.001
            )
            grad_dense_1, grad_bias_hidden, _ = back.backpropagate(
                dense_layer_1, batch_relu_2, hidden_errors, actual_batch_size
            )
            pooled_errors = (hidden_errors @ dense_layer_1.T).reshape(
                actual_batch_size, filter_count, 13, 13
            ) * back.deriv_leaky_RELU(
                batch_pooled, 0.01 if model == "v2" else 0.001
            )

            pool_mask = back_propagation.deriv_max_pooling(batch_activated)
            activated_errors = (
                cp.repeat(cp.repeat(pooled_errors, 2, axis=2), 2, axis=3)
                * pool_mask
            )
            conv_errors = activated_errors * back.deriv_leaky_RELU(
                batch_conv, 0.001
            )

            if model == "v2":
                _, grad_filter, _ = back.conv_backward_im2col(
                    conv_errors,
                    (
                        batch_images,
                        conv_filter,
                        None,
                        {"stride": 1, "pad": 0},
                    ),
                )
                grad_filter /= actual_batch_size
                grad_dense_2 = cp.clip(grad_dense_2, -1.0, 1.0)
                grad_dense_1 = cp.clip(grad_dense_1, -1.0, 1.0)
                grad_filter = cp.clip(grad_filter, -1.0, 1.0)
                dense_layer_2 = optimizer.update("W2", dense_layer_2, grad_dense_2)
                bias_out = optimizer.update("b2", bias_out, grad_bias_out)
                dense_layer_1 = optimizer.update("W1", dense_layer_1, grad_dense_1)
                bias_hidden = optimizer.update("b1", bias_hidden, grad_bias_hidden)
                conv_filter = optimizer.update("conv_filter", conv_filter, grad_filter)
            else:
                grad_filter, _ = back.deriv_conv_layer(
                    batch_images, conv_filter[0], batch_conv[:, 0], conv_errors[:, 0]
                )
                dense_layer_2 -= learning_rate * cp.clip(grad_dense_2, -1e5, 1e5)
                bias_out -= learning_rate * cp.clip(grad_bias_out, -1e5, 1e5)
                dense_layer_1 -= learning_rate * cp.clip(grad_dense_1, -1e5, 1e5)
                bias_hidden -= learning_rate * cp.clip(
                    grad_bias_hidden, -1e5, 1e5
                )
                conv_filter[0] -= learning_rate * grad_filter

            batch_loss = back.cross_entropy_loss(probabilities, batch_labels)
            total_loss += float(cp.sum(batch_loss).get())

        epoch_loss = total_loss / sample_count
        validation_loss_total = 0.0
        validation_correct = 0
        for start in range(0, validation_count, batch_size):
            batch_end = min(start + batch_size, validation_count)
            probabilities = _predict_probabilities(
                model,
                validation_data[start:batch_end],
                conv_filter,
                dense_layer_1,
                bias_hidden,
                dense_layer_2,
                bias_out,
            )
            labels = validation_labels[start:batch_end]
            validation_losses = back.cross_entropy_loss(probabilities, labels)
            validation_loss_total += float(cp.sum(validation_losses).get())
            validation_correct += int(cp.sum(cp.argmax(probabilities, axis=1) == labels).get())

        epoch_validation_loss = validation_loss_total / validation_count
        epoch_validation_accuracy = validation_correct / validation_count
        history.append(epoch_loss)
        validation_history.append(epoch_validation_loss)
        validation_accuracy.append(epoch_validation_accuracy)
        on_epoch(
            {
                "epoch": epoch + 1,
                "epochs": epochs,
                "loss": epoch_loss,
                "validation_loss": epoch_validation_loss,
                "validation_accuracy": epoch_validation_accuracy,
            }
        )

    preview = _create_preview(
        model,
        training_data[0],
        int(training_labels[0].get()),
        conv_filter,
        dense_layer_1,
        bias_hidden,
        dense_layer_2,
        bias_out,
    )
    return {
        "history": history,
        "validation_history": validation_history,
        "validation_accuracy": validation_accuracy,
        "preview": preview,
    }
