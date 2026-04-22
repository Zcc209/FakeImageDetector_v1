const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("resultBox");

analyzeBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    alert("Please select an image first.");
    return;
  }

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  resultBox.textContent = "Analyzing...";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: fd,
    });

    const data = await res.json();
    if (!res.ok) {
      resultBox.textContent = `HTTP ${res.status}\n` + JSON.stringify(data, null, 2);
      return;
    }

    resultBox.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    resultBox.textContent =
      "Network error: cannot reach backend. Start Flask first and open http://localhost:8000\n" +
      String(err);
  }
});
