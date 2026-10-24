import logging
import math
import threading
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for

from demo import train_model

app = Flask(__name__)
training_lock = threading.Lock()
training_state: dict[str, Any] = {
    "status": "idle",
    "epoch": 0,
    "epochs": 0,
    "history": [],
    "validation_history": [],
    "validation_accuracy": [],
}


def _run_training(settings: dict[str, Any]) -> None:
    def record_epoch(progress: dict[str, object]) -> None:
        with training_lock:
            training_state["epoch"] = progress["epoch"]
            training_state["epochs"] = progress["epochs"]
            training_state["history"].append(progress["loss"])
            training_state["validation_history"].append(progress["validation_loss"])
            training_state["validation_accuracy"].append(
                progress["validation_accuracy"]
            )

    try:
        result = train_model(
            model=settings["model"],
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            sample_count=settings["sample_count"],
            validation_count=settings["validation_count"],
            learning_rate=settings["learning_rate"],
            on_epoch=record_epoch,
        )
    except Exception as error:
        app.logger.exception("Demo model training failed")
        with training_lock:
            training_state.update(status="error", error=str(error))
        return

    with training_lock:
        training_state.update(
            status="complete",
            history=result["history"],
            validation_history=result["validation_history"],
            validation_accuracy=result["validation_accuracy"],
            preview=result["preview"],
        )


@app.route("/home")
def home():
    return render_template("main.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


@app.route("/api/training", methods=["GET", "POST"])
def training():
    if request.method == "GET":
        with training_lock:
            return jsonify(training_state.copy())

    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify(error="Send the training settings as a JSON object."), 400

    model = payload.get("model")
    if model not in {"v1", "v2"}:
        return jsonify(error="Choose either model V1 or V2."), 400

    max_epochs = 200 if model == "v1" else 20
    int_settings = {
        "epochs": (payload.get("epochs"), 1, max_epochs),
        "batch_size": (payload.get("batch_size"), 1, 128),
        "sample_count": (payload.get("sample_count"), 1, 2048),
        "validation_count": (payload.get("validation_count"), 1, 2048),
    }
    settings: dict[str, Any] = {"model": model}
    for name, (value, minimum, maximum) in int_settings.items():
        if type(value) is not int or not minimum <= value <= maximum:
            return (
                jsonify(
                    error=f"{name.replace('_', ' ').capitalize()} must be between "
                    f"{minimum} and {maximum}."
                ),
                400,
            )
        settings[name] = value

    learning_rate = payload.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or not 0.00001 <= learning_rate <= 1
    ):
        return jsonify(error="Learning rate must be between 0.00001 and 1."), 400
    settings["learning_rate"] = float(learning_rate)

    with training_lock:
        if training_state["status"] == "running":
            return jsonify(error="A training run is already in progress."), 409
        training_state.clear()
        training_state.update(
            status="running",
            epoch=0,
            epochs=settings["epochs"],
            history=[],
            validation_history=[],
            validation_accuracy=[],
            model=model,
        )

    worker = threading.Thread(
        target=_run_training, args=(settings,), name="demo-model-training", daemon=True
    )
    worker.start()
    return jsonify(status="running"), 202


@app.route("/")
def page():
    return redirect(url_for("home"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, use_reloader=False)