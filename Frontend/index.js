const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("UploadFileButton");
const removeButton = document.getElementById("RemoveFileButton");
const loopCountInput = document.getElementById("loopCountInput");
const plotPathInput = document.getElementById("plotPathInput");

const status = document.getElementById("status");
const dropZone = document.getElementById("dropZone");
const dropZoneFileName = document.getElementById("dropZoneFileName");
const dropZoneEmpty = document.querySelector(".drop-zone-empty");
const dropZoneLoaded = document.querySelector(".drop-zone-loaded");

const resultsSection = document.getElementById("resultsSection");
const resultsLayer = document.querySelector(".results-layer");
const metaFilename = document.getElementById("metaFilename");
const metaVectorLength = document.getElementById("metaVectorLength");
const metaLoopCount = document.getElementById("metaLoopCount");
const iterationsContainer = document.getElementById("iterationsContainer");

const planetMain = document.getElementById("planetMain");
const planetAtmosphere = document.getElementById("planetAtmosphere");
const planetRim = document.getElementById("planetRim");

const FEATURE_COUNT = 17;

const FEATURE_NAMES = [
    "Planet radius",
    "Planet density (g/cm3)",
    "Planet surface pressure (bar)",
    "Planet surface temperature (Kelvin)",
    "H2O abundance",
    "CO2 abundance",
    "O2 abundance",
    "N2 abundance",
    "CH4 abundance",
    "N2O abundance",
    "CO abundance",
    "O3 abundance",
    "SO2 abundance",
    "NH3 abundance",
    "C2H6 abundance",
    "NO2 abundance",
    "Planet's mean surface albedo"
];

let selectedFile = null;
let loopCount = null;
let plotPath = "";

let currentPlanetState = {
    mainScale: 0.72,
    atmoScale: 0.72,
    rimScale: 0.72,
    mainY: 0,
    atmoY: 0,
    rimY: 0,
};

let targetPlanetState = {
    mainScale: 0.72,
    atmoScale: 0.72,
    rimScale: 0.72,
    mainY: 0,
    atmoY: 0,
    rimY: 0,
};

let animationFrameId = null;

function lerp(start, end, amount) {
    return start + (end - start) * amount;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function parseLoopCount(rawValue) {
    if (rawValue === "" || rawValue === null || rawValue === undefined) {
        return null;
    }

    const numericValue = Number(rawValue);

    if (!Number.isFinite(numericValue)) {
        return null;
    }

    return Math.round(numericValue);
}

function normalizeLoopCount(rawValue) {
    const parsedValue = parseLoopCount(rawValue);

    if (parsedValue === null) {
        return null;
    }

    return clamp(parsedValue, 2, 64);
}

function normalizePlotPath(rawValue) {
    if (typeof rawValue !== "string") {
        return "";
    }

    return rawValue.trim();
}

function hasValidLoopCount() {
    return Number.isInteger(loopCount) && loopCount >= 2 && loopCount <= 64;
}

function hasValidPlotPath() {
    if (!plotPath) {
        return false;
    }

    const lowerPath = plotPath.toLowerCase();
    return lowerPath.endsWith(".png") || lowerPath.endsWith(".jpg") || lowerPath.endsWith(".jpeg");
}

function updateAnalyzeButtonState() {
    uploadButton.disabled = !(selectedFile && hasValidLoopCount() && hasValidPlotPath());
}

function updateDropZoneState() {
    if (selectedFile) {
        dropZone.classList.add("has-file");
        dropZoneEmpty.hidden = true;
        dropZoneLoaded.hidden = false;
        dropZoneFileName.textContent = selectedFile.name;
    } else {
        dropZone.classList.remove("has-file");
        dropZoneEmpty.hidden = false;
        dropZoneLoaded.hidden = true;
        dropZoneFileName.textContent = "";
    }

    updateAnalyzeButtonState();
}

function updateLoopUI() {
    const valid = hasValidLoopCount();

    if (loopCount === null) {
        loopCountInput.classList.remove("loop-invalid");
    } else if (valid) {
        loopCountInput.classList.remove("loop-invalid");
    } else {
        loopCountInput.classList.add("loop-invalid");
    }

    updateAnalyzeButtonState();
}

function updatePlotPathUI() {
    if (plotPath === "") {
        plotPathInput.classList.remove("loop-invalid");
    } else if (hasValidPlotPath()) {
        plotPathInput.classList.remove("loop-invalid");
    } else {
        plotPathInput.classList.add("loop-invalid");
    }

    updateAnalyzeButtonState();
}

function updateStatusAfterInputChange() {
    if (!hasValidPlotPath()) {
        status.textContent = "Set a valid plot output path ending with .png or .jpg.";
        return;
    }

    if (!selectedFile) {
        status.textContent = "Plot output path is set. Add a spectrum file to continue.";
        return;
    }

    if (!hasValidLoopCount()) {
        status.textContent = "Set model repeats from 2 to 64 to enable the analysis.";
        return;
    }

    status.textContent = "File, plot path and repeat count are ready for upload.";
}

function setSelectedFile(file) {
    selectedFile = file;
    updateDropZoneState();
    updateStatusAfterInputChange();
}

function clearSelectedFile(message = "File removed.") {
    selectedFile = null;
    fileInput.value = "";
    updateDropZoneState();
    status.textContent = message;
}

function setPlotPath(rawValue, { commitToInput = false } = {}) {
    plotPath = normalizePlotPath(rawValue);

    if (commitToInput) {
        plotPathInput.value = plotPath;
    }

    updatePlotPathUI();
    updateStatusAfterInputChange();
}

function setLoopCount(rawValue, { commitToInput = false, clampValue = false } = {}) {
    const parsed = clampValue ? normalizeLoopCount(rawValue) : parseLoopCount(rawValue);
    loopCount = parsed;

    if (commitToInput) {
        loopCountInput.value = parsed === null ? "" : String(parsed);
    }

    updateLoopUI();
    updateStatusAfterInputChange();
}

function formatValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "—";
    }

    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    if (numericValue === 0) {
        return "0";
    }

    const absValue = Math.abs(numericValue);

    if (absValue >= 1000 || absValue < 0.0001) {
        return numericValue.toExponential(4);
    }

    return numericValue.toFixed(6).replace(/\.?0+$/, "");
}

function clearResultCells() {
    for (let i = 0; i < FEATURE_COUNT; i++) {
        const predCell = document.getElementById(`pred-${i}`);
        const stdCell = document.getElementById(`std-${i}`);
        const errCell = document.getElementById(`err-${i}`);

        if (predCell) predCell.textContent = "—";
        if (stdCell) stdCell.textContent = "—";
        if (errCell) errCell.textContent = "—";
    }
}

function clearIterationTables() {
    if (!iterationsContainer) return;

    iterationsContainer.innerHTML = `
        <p class="iterations-placeholder">Run analysis to inspect individual iterations.</p>
    `;
}

function renderIterationTables(predictions, means = []) {
    if (!iterationsContainer) return;

    if (!Array.isArray(predictions) || predictions.length === 0) {
        clearIterationTables();
        return;
    }

    const cardsHtml = predictions.map((iterationValues, iterationIndex) => {
        const rowsHtml = FEATURE_NAMES.map((featureName, featureIndex) => {
            const iterationValue = Array.isArray(iterationValues) ? iterationValues[featureIndex] : null;
            const meanValue = means[featureIndex];
            const deltaValue =
                Number.isFinite(Number(iterationValue)) && Number.isFinite(Number(meanValue))
                    ? Number(iterationValue) - Number(meanValue)
                    : null;

            return `
                <div class="iteration-row">
                    <span>${featureName}</span>
                    <span>${formatValue(iterationValue)}</span>
                    <span>${formatValue(deltaValue)}</span>
                </div>
            `;
        }).join("");

        return `
            <div class="iteration-card">
                <div class="iteration-card-header">
                    <div>
                        <div class="iteration-title">Iteration ${iterationIndex + 1}</div>
                        <div class="iteration-subtitle">
                            Compare this repeat against the aggregated result above.
                        </div>
                    </div>
                </div>

                <div class="iteration-grid-head">
                    <span>Feature</span>
                    <span>Prediction</span>
                    <span>Δ vs Mean</span>
                </div>

                <div class="iteration-grid-body">
                    ${rowsHtml}
                </div>
            </div>
        `;
    }).join("");

    iterationsContainer.innerHTML = cardsHtml;
}

function renderResults(data) {
    const means = Array.isArray(data.mean) ? data.mean : [];
    const stds = Array.isArray(data.std) ? data.std : [];
    const errors = Array.isArray(data.errors) ? data.errors : [];
    const predictions = Array.isArray(data.predicitons) ? data.predicitons : [];

    for (let i = 0; i < FEATURE_COUNT; i++) {
        const predCell = document.getElementById(`pred-${i}`);
        const stdCell = document.getElementById(`std-${i}`);
        const errCell = document.getElementById(`err-${i}`);

        if (predCell) predCell.textContent = formatValue(means[i]);
        if (stdCell) stdCell.textContent = formatValue(stds[i]);
        if (errCell) errCell.textContent = formatValue(errors[i]);
    }

    metaFilename.textContent = data.filename || selectedFile?.name || "Demo spectrum";
    metaVectorLength.textContent = Array.isArray(predictions?.[0])
        ? predictions[0].length
        : "—";
    metaLoopCount.textContent = data.num_repeats ?? loopCount ?? "—";

    renderIterationTables(predictions, means);
}

async function uploadSelectedFile() {
    if (!selectedFile) {
        status.textContent = "Please select a file first.";
        return;
    }

    if (!hasValidPlotPath()) {
        status.textContent = "Please set a valid plot output path ending with .png or .jpg first.";
        plotPathInput.focus();
        return;
    }

    loopCount = normalizeLoopCount(loopCountInput.value);
    loopCountInput.value = loopCount === null ? "" : String(loopCount);
    updateLoopUI();

    if (!hasValidLoopCount()) {
        status.textContent = "Please set model repeats from 2 to 64 first.";
        loopCountInput.focus();
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("num_repeats", String(loopCount));
    formData.append("plot_path", plotPath);

    uploadButton.disabled = true;
    status.textContent = "Uploading and generating analysis...";

    try {
        const response = await fetch("http://127.0.0.1:2137/upload/", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            status.textContent = data.detail || "Upload failed.";
            metaFilename.textContent = "Awaiting upload";
            metaVectorLength.textContent = "—";
            metaLoopCount.textContent = "—";
            clearResultCells();
            clearIterationTables();
            updateAnalyzeButtonState();
            return;
        }

        renderResults(data);
        status.textContent = "Analysis completed successfully.";
        updateAnalyzeButtonState();
    } catch (error) {
        console.error(error);
        status.textContent = "Could not connect to the backend.";
        metaFilename.textContent = "Awaiting upload";
        metaVectorLength.textContent = "—";
        metaLoopCount.textContent = "—";
        clearResultCells();
        clearIterationTables();
        updateAnalyzeButtonState();
    }
}

function calculatePlanetTarget() {
    const rect = resultsSection.getBoundingClientRect();
    const viewportHeight = window.innerHeight;

    const rawProgress = (viewportHeight - rect.top + viewportHeight * 0.12) / (viewportHeight + rect.height * 0.22);
    const progress = clamp(rawProgress, 0, 1);

    targetPlanetState.mainScale = 0.72 + progress * 0.78;
    targetPlanetState.atmoScale = 0.72 + progress * 0.86;
    targetPlanetState.rimScale = 0.72 + progress * 0.82;

    targetPlanetState.mainY = progress * -12;
    targetPlanetState.atmoY = progress * -16;
    targetPlanetState.rimY = progress * -14;
}

function renderPlanetFrame() {
    currentPlanetState.mainScale = lerp(currentPlanetState.mainScale, targetPlanetState.mainScale, 0.08);
    currentPlanetState.atmoScale = lerp(currentPlanetState.atmoScale, targetPlanetState.atmoScale, 0.08);
    currentPlanetState.rimScale = lerp(currentPlanetState.rimScale, targetPlanetState.rimScale, 0.08);

    currentPlanetState.mainY = lerp(currentPlanetState.mainY, targetPlanetState.mainY, 0.08);
    currentPlanetState.atmoY = lerp(currentPlanetState.atmoY, targetPlanetState.atmoY, 0.08);
    currentPlanetState.rimY = lerp(currentPlanetState.rimY, targetPlanetState.rimY, 0.08);

    planetMain.style.transform = `translate3d(0, ${currentPlanetState.mainY}vh, 0) scale(${currentPlanetState.mainScale})`;
    planetAtmosphere.style.transform = `translate3d(0, ${currentPlanetState.atmoY}vh, 0) scale(${currentPlanetState.atmoScale})`;
    planetRim.style.transform = `translate3d(0, ${currentPlanetState.rimY}vh, 0) scale(${currentPlanetState.rimScale})`;

    const layerRise = currentPlanetState.mainY * 0.55;
    const layerScale = 0.985 + (currentPlanetState.mainScale - 0.72) * 0.03;

    resultsLayer.style.transform = `translate3d(0, ${layerRise}vh, 0) scale(${layerScale})`;
    resultsLayer.style.transformOrigin = "center top";

    const done =
        Math.abs(currentPlanetState.mainScale - targetPlanetState.mainScale) < 0.001 &&
        Math.abs(currentPlanetState.atmoScale - targetPlanetState.atmoScale) < 0.001 &&
        Math.abs(currentPlanetState.rimScale - targetPlanetState.rimScale) < 0.001 &&
        Math.abs(currentPlanetState.mainY - targetPlanetState.mainY) < 0.01 &&
        Math.abs(currentPlanetState.atmoY - targetPlanetState.atmoY) < 0.01 &&
        Math.abs(currentPlanetState.rimY - targetPlanetState.rimY) < 0.01;

    if (!done) {
        animationFrameId = requestAnimationFrame(renderPlanetFrame);
    } else {
        animationFrameId = null;
    }
}

function requestPlanetUpdate() {
    calculatePlanetTarget();

    if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(renderPlanetFrame);
    }
}

uploadButton.addEventListener("click", uploadSelectedFile);

removeButton.addEventListener("click", () => {
    clearSelectedFile("");
    plotPath = "";
    plotPathInput.value = "";
    plotPathInput.classList.remove("loop-invalid");
    updatePlotPathUI();

    loopCount = null;
    loopCountInput.value = "";
    loopCountInput.classList.remove("loop-invalid");
    updateLoopUI();
    clearResultCells();
    clearIterationTables();

    metaFilename.textContent = "Awaiting upload";
    metaVectorLength.textContent = "—";
    metaLoopCount.textContent = "—";

    status.textContent = "File, repeat count and plot path removed.";
});

loopCountInput.addEventListener("input", (event) => {
    const value = event.target.value;

    if (value === "") {
        loopCount = null;
        loopCountInput.classList.remove("loop-invalid");
        updateAnalyzeButtonState();
        updateStatusAfterInputChange();
        return;
    }

    setLoopCount(value, { commitToInput: false, clampValue: false });
});

loopCountInput.addEventListener("blur", (event) => {
    const value = event.target.value;

    if (value === "") {
        loopCount = null;
        loopCountInput.classList.remove("loop-invalid");
        updateAnalyzeButtonState();
        updateStatusAfterInputChange();
        return;
    }

    setLoopCount(value, { commitToInput: false, clampValue: false });
});

plotPathInput.addEventListener("input", (event) => {
    setPlotPath(event.target.value, { commitToInput: false });
});

plotPathInput.addEventListener("blur", (event) => {
    setPlotPath(event.target.value, { commitToInput: true });
});

fileInput.addEventListener("change", () => {
    const files = fileInput.files;

    if (!files || files.length === 0) {
        clearSelectedFile("No file selected.");
        return;
    }

    if (files.length > 1) {
        clearSelectedFile("Only one file is allowed.");
        return;
    }

    setSelectedFile(files[0]);
});

dropZone.addEventListener("click", () => {
    fileInput.click();
});

dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-active");
    });
});

["dragleave", "dragend"].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove("drag-active");
    });
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-active");

    const files = event.dataTransfer.files;

    if (!files || files.length === 0) {
        status.textContent = "No file detected.";
        return;
    }

    if (files.length > 1) {
        status.textContent = "Only one file is allowed.";
        return;
    }

    setSelectedFile(files[0]);
});

window.addEventListener("scroll", requestPlanetUpdate, { passive: true });
window.addEventListener("resize", requestPlanetUpdate);
window.addEventListener("load", requestPlanetUpdate);

updateDropZoneState();
updateLoopUI();
updatePlotPathUI();
clearIterationTables();
requestPlanetUpdate();