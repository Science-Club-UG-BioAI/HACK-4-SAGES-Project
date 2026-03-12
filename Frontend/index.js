
const btn = document.getElementById("btn");
const result = document.getElementById("result");

btn.addEventListener("click", async () => {
    const response = await fetch("/test");
    const data = await response.json();
    result.textContent = data.message;
});

const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("UploadFileButton");
const status = document.getElementById("status");
const output = document.getElementById("output");

uploadButton.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        status.textContent = "Please select a file.";
        return;
    }
    const formData = new FormData();
    formData.append("file", file);
    status.textContent = "Uploading";
    output.textContent = "";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        
        if (!response.ok) {
            status.textContent = "Error (with file)";
            output.textContent = data.detail || "67 67";
            return;
        }
        status.textContent = "Success";
        output.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        status.textContent = "Error (backend)";
        output.textContent = String(error);
    }
});
