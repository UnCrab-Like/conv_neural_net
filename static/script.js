const form = document.querySelector("#training-form");
const trainButton = document.querySelector("#train-button");
const statusMessage = document.querySelector("#status-message");
const epochCount = document.querySelector("#epoch-count");
const chart = document.querySelector("#loss-chart");
const errorMessage = document.querySelector("#error-message");
const visualizationPanel = document.querySelector("#visualization-panel");
const imageSteps = document.querySelector("#image-steps");
const predictionMessage = document.querySelector("#prediction-message");
const epochInput = form.elements.namedItem("epochs");
const epochLimit = document.querySelector("#epoch-limit");

function updateEpochLimit() {
    const selectedModel = form.elements.namedItem("model").value;
    const maxEpochs = selectedModel === "v1" ? 200 : 20;
    epochInput.max = String(maxEpochs);
    epochLimit.textContent = `${selectedModel.toUpperCase()} supports up to ${maxEpochs} epochs`;
    if (Number(epochInput.value) > maxEpochs) {
        epochInput.value = String(maxEpochs);
    }
}

function renderChart(history, validationHistory) {
    chart.replaceChildren();
    if (history.length === 0) {
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", "300");
        label.setAttribute("y", "110");
        label.setAttribute("text-anchor", "middle");
        label.textContent = "Loss chart will appear here";
        chart.append(label);
        return;
    }

    const width = 600;
    const height = 240;
    const padding = { top: 20, right: 24, bottom: 52, left: 70 };
    const allLosses = [...history, ...validationHistory];
    const largestLoss = Math.max(...allLosses, 0.01);
    const rawStep = largestLoss / 4;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalizedStep = rawStep / magnitude;
    const tickStep = (normalizedStep <= 1 ? 1
        : normalizedStep <= 2 ? 2
            : normalizedStep <= 5 ? 5 : 10) * magnitude;
    const maxLoss = Math.ceil(largestLoss / tickStep) * tickStep;
    const x = (index) => padding.left +
        (history.length === 1
            ? (width - padding.left - padding.right) / 2
            : index * (width - padding.left - padding.right) / (history.length - 1));
    const y = (loss) => padding.top +
        (maxLoss - loss) * (height - padding.top - padding.bottom) / maxLoss;

    for (let value = 0; value <= maxLoss + tickStep * 0.001; value += tickStep) {
        const yPosition = y(value);
        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", String(padding.left));
        grid.setAttribute("x2", String(width - padding.right));
        grid.setAttribute("y1", String(yPosition));
        grid.setAttribute("y2", String(yPosition));
        grid.setAttribute("class", "chart-grid");
        chart.append(grid);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(padding.left - 10));
        label.setAttribute("y", String(yPosition + 4));
        label.setAttribute("text-anchor", "end");
        label.textContent = value.toFixed(tickStep < 1 ? 1 : 0);
        chart.append(label);
    }

    const tickInterval = Math.max(1, Math.ceil(history.length / 6));
    const tickIndices = new Set();
    for (let index = 0; index < history.length; index += tickInterval) {
        tickIndices.add(index);
    }
    tickIndices.add(history.length - 1);

    const appendSeries = (values, className) => {
        if (values.length === 0) return;
        const seriesLine = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        seriesLine.setAttribute(
            "points",
            values.map((loss, index) => `${x(index)},${y(loss)}`).join(" ")
        );
        seriesLine.setAttribute("class", className);
        chart.append(seriesLine);
    };
    appendSeries(history, "loss-line");
    appendSeries(validationHistory, "validation-loss-line");

    history.forEach((loss, index) => {
        const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        point.setAttribute("cx", String(x(index)));
        point.setAttribute("cy", String(y(loss)));
        point.setAttribute("r", "4");
        point.setAttribute("class", "loss-point training-loss-point");
        chart.append(point);
    });
    validationHistory.forEach((loss, index) => {
        const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        point.setAttribute("cx", String(x(index)));
        point.setAttribute("cy", String(y(loss)));
        point.setAttribute("r", "4");
        point.setAttribute("class", "loss-point validation-loss-point");
        chart.append(point);
    });

    tickIndices.forEach((index) => {
        const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
        tick.setAttribute("x1", String(x(index)));
        tick.setAttribute("x2", String(x(index)));
        tick.setAttribute("y1", String(height - padding.bottom));
        tick.setAttribute("y2", String(height - padding.bottom + 5));
        tick.setAttribute("class", "chart-grid");
        chart.append(tick);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(x(index)));
        label.setAttribute("y", String(height - padding.bottom + 19));
        label.setAttribute("text-anchor", "middle");
        label.textContent = String(index + 1);
        chart.append(label);
    });

    const axisLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    axisLabel.setAttribute("x", "16");
    axisLabel.setAttribute("y", String((height - padding.bottom + padding.top) / 2));
    axisLabel.setAttribute("text-anchor", "middle");
    axisLabel.setAttribute("transform", `rotate(-90 16 ${axisLabel.getAttribute("y")})`);
    axisLabel.textContent = "Loss";
    chart.append(axisLabel);
}

function renderPreview(preview) {
    if (!preview) {
        visualizationPanel.hidden = true;
        return;
    }

    predictionMessage.textContent =
        `Actual digit: ${preview.label} · Predicted: ${preview.prediction} · ` +
        `${(preview.confidence * 100).toFixed(1)}% confidence`;
    imageSteps.replaceChildren();
    Object.entries(preview.images).forEach(([title, src]) => {
        const item = document.createElement("div");
        item.className = "image-step";
        const image = document.createElement("img");
        image.src = src;
        image.alt = `${title} visualization`;
        const caption = document.createElement("strong");
        caption.textContent = title;
        item.append(image, caption);
        imageSteps.append(item);
    });
    visualizationPanel.hidden = false;
}

function updateProgress(state) {
    const history = state.history || [];
    const validationHistory = state.validation_history || [];
    renderChart(history, validationHistory);
    epochCount.textContent = history.length
        ? `Epoch ${state.epoch} of ${state.epochs}`
        : "No epochs yet";
    if (state.status === "running") {
        if (validationHistory.length) {
            const latestLoss = validationHistory.at(-1);
            const latestAccuracy = state.validation_accuracy.at(-1);
            statusMessage.textContent =
                `Training ${state.model.toUpperCase()}… Validation loss: ` +
                `${latestLoss.toFixed(3)} · accuracy: ${(latestAccuracy * 100).toFixed(1)}%`;
        } else {
            statusMessage.textContent = `Training ${state.model.toUpperCase()}…`;
        }
    } else if (state.status === "complete") {
        const latestAccuracy = state.validation_accuracy?.at(-1);
        statusMessage.textContent = latestAccuracy === undefined
            ? "Training complete."
            : `Training complete. Validation accuracy: ${(latestAccuracy * 100).toFixed(1)}%.`;
        trainButton.disabled = false;
        renderPreview(state.preview);
    } else if (state.status === "error") {
        statusMessage.textContent = "Training stopped.";
        trainButton.disabled = false;
        errorMessage.textContent = state.error || "Training failed.";
        errorMessage.hidden = false;
    }
}

async function pollProgress() {
    try {
        const response = await fetch("/api/training");
        if (!response.ok) {
            throw new Error(`Could not read training progress (${response.status}).`);
        }
        const state = await response.json();
        updateProgress(state);
        if (state.status === "running") {
            window.setTimeout(pollProgress, 600);
        }
    } catch (error) {
        statusMessage.textContent = "Could not read training progress.";
        errorMessage.textContent = error.message;
        errorMessage.hidden = false;
        trainButton.disabled = false;
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorMessage.hidden = true;
    visualizationPanel.hidden = true;
    renderChart([], []);
    const formData = new FormData(form);
    const settings = {
        model: formData.get("model"),
        epochs: Number(formData.get("epochs")),
        batch_size: Number(formData.get("batch_size")),
        sample_count: Number(formData.get("sample_count")),
        validation_count: Number(formData.get("validation_count")),
        learning_rate: Number(formData.get("learning_rate")),
    };
    trainButton.disabled = true;
    statusMessage.textContent = "Starting training…";

    try {
        const response = await fetch("/api/training", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || `Could not start training (${response.status}).`);
        }
        pollProgress();
    } catch (error) {
        statusMessage.textContent = "Training could not be started.";
        errorMessage.textContent = error.message;
        errorMessage.hidden = false;
        trainButton.disabled = false;
    }
});

form.querySelectorAll('input[name="model"]').forEach((radio) => {
    radio.addEventListener("change", updateEpochLimit);
});
updateEpochLimit();
renderChart([], []);