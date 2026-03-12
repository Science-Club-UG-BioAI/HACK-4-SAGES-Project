const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("UploadFileButton");
const removeButton = document.getElementById("RemoveFileButton");
const loopCountInput = document.getElementById("loopCountInput");

const status = document.getElementById("status");
const dropZone = document.getElementById("dropZone");
const dropZoneFileName = document.getElementById("dropZoneFileName");
const dropZoneEmpty = document.querySelector(".drop-zone-empty");
const dropZoneLoaded = document.querySelector(".drop-zone-loaded");

const resultsSection = document.getElementById("resultsSection");
const resultsLayer = document.querySelector(".results-layer");
const metaFilename = document.getElementById("metaFilename");
const metaVectorLength = document.getElementById("metaVectorLength");
const metaFirstValues = document.getElementById("metaFirstValues");
const metaLoopCount = document.getElementById("metaLoopCount");

const planetMain = document.getElementById("planetMain");
const planetAtmosphere = document.getElementById("planetAtmosphere");
const planetRim = document.getElementById("planetRim");

const FEATURE_COUNT = 17;

let selectedFile = null;
let loopCount = null;

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

function normalizeLoopCount(rawValue) {
    if (rawValue === "" || rawValue === null || rawValue === undefined) {
        return null;
    }

    const numericValue = Number(rawValue);

    if (!Number.isFinite(numericValue)) {
        return null;
    }

    const integerValue = Math.round(numericValue);
    return clamp(integerValue, 2, 64);
}

function hasValidLoopCount() {
    return Number.isInteger(loopCount) && loopCount >= 2 && loopCount <= 64;
}

function updateAnalyzeButtonState() {
    uploadButton.disabled = !(selectedFile && hasValidLoopCount());
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
        loopCountInput.value = String(loopCount);
    } else {
        loopCountInput.classList.add("loop-invalid");
    }

    updateAnalyzeButtonState();
}

function setSelectedFile(file) {
    selectedFile = file;
    updateDropZoneState();

    if (selectedFile && !hasValidLoopCount()) {
        status.textContent = "Set model repeats from 2 to 64 to enable the analysis.";
    } else if (selectedFile && hasValidLoopCount()) {
        status.textContent = "File and repeat count are ready for upload.";
    }
}

function clearSelectedFile(message = "File removed.") {
    selectedFile = null;
    fileInput.value = "";
    updateDropZoneState();
    status.textContent = message;
}

function setLoopCount(rawValue, { commitToInput = false } = {}) {
    const normalized = normalizeLoopCount(rawValue);
    loopCount = normalized;

    if (commitToInput && normalized !== null) {
        loopCountInput.value = String(normalized);
    }

    if (commitToInput && normalized === null) {
        loopCountInput.value = "";
    }

    updateLoopUI();

    if (!selectedFile && hasValidLoopCount()) {
        status.textContent = "Repeat count is set. Add a spectrum file to continue.";
    } else if (selectedFile && hasValidLoopCount()) {
        status.textContent = "File and repeat count are ready for upload.";
    } else if (selectedFile && !hasValidLoopCount()) {
        status.textContent = "Set model repeats from 2 to 64 to enable the analysis.";
    }
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

function renderResults(data) {
    const means = Array.isArray(data.mean) ? data.mean : [];
    const stds = Array.isArray(data.std) ? data.std : [];
    const errors = Array.isArray(data.errors) ? data.errors : [];

    for (let i = 0; i < FEATURE_COUNT; i++) {
        const predCell = document.getElementById(`pred-${i}`);
        const stdCell = document.getElementById(`std-${i}`);
        const errCell = document.getElementById(`err-${i}`);

        if (predCell) predCell.textContent = formatValue(means[i]);
        if (stdCell) stdCell.textContent = formatValue(stds[i]);
        if (errCell) errCell.textContent = formatValue(errors[i]);
    }

    metaFilename.textContent = data.filename || selectedFile?.name || "Demo spectrum";
    metaVectorLength.textContent = Array.isArray(data.predicitons?.[0])
        ? data.predicitons[0].length
        : "—";
    metaFirstValues.textContent = means.length > 0
        ? means.slice(0, 3).map(formatValue).join(", ")
        : "—";
    metaLoopCount.textContent = data.num_repeats ?? loopCount ?? "—";
}

async function uploadSelectedFile() {
    if (!selectedFile) {
        status.textContent = "Please select a file first.";
        return;
    }

    if (!hasValidLoopCount()) {
        status.textContent = "Please set model repeats from 2 to 64 first.";
        loopCountInput.focus();
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("num_repeats", String(loopCount));
    formData.append("plot_path", "out/plot.png")

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
            metaFirstValues.textContent = "—";
            metaLoopCount.textContent = "—";
            clearResultCells();
            updateAnalyzeButtonState();
            return;
        }

        renderResults(data);
        status.textContent = "Spectrum processed successfully.";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        console.log(error)
        status.textContent = "Backend connection error.";
        metaFilename.textContent = "Awaiting upload";
        metaVectorLength.textContent = "—";
        metaFirstValues.textContent = "—";
        metaLoopCount.textContent = "—";
        clearResultCells();
    } finally {
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
    loopCount = null;
    loopCountInput.value = "";
    loopCountInput.classList.remove("loop-invalid");
    updateLoopUI();
    clearResultCells();

    metaFilename.textContent = "Awaiting upload";
    metaVectorLength.textContent = "—";
    metaFirstValues.textContent = "—";
    metaLoopCount.textContent = "—";

    status.textContent = "File and repeat count removed.";
});

loopCountInput.addEventListener("input", (event) => {
    const value = event.target.value;

    if (value === "") {
        loopCount = null;
        loopCountInput.classList.remove("loop-invalid");
        updateAnalyzeButtonState();

        if (selectedFile) {
            status.textContent = "Set model repeats from 2 to 64 to enable the analysis.";
        }
        return;
    }

    setLoopCount(value, { commitToInput: false });
});

loopCountInput.addEventListener("blur", (event) => {
    const value = event.target.value;

    if (value === "") {
        loopCount = null;
        loopCountInput.classList.remove("loop-invalid");
        updateAnalyzeButtonState();
        return;
    }

    setLoopCount(value, { commitToInput: true });
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
requestPlanetUpdate();