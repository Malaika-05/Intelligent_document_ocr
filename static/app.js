const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const fileInput = document.getElementById("fileInput");
const scanFrame = document.getElementById("scanFrame");
const previewImg = document.getElementById("previewImg");
const scanLine = document.getElementById("scanLine");

const statusRow = document.getElementById("statusRow");
const statusText = document.getElementById("statusText");

const resultsEl = document.getElementById("results");
const objectChips = document.getElementById("objectChips");
const printedText = document.getElementById("printedText");
const printedCount = document.getElementById("printedCount");
const handwrittenText = document.getElementById("handwrittenText");
const indexNote = document.getElementById("indexNote");

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");

let documentReady = false;

// ---------------- Upload handling ----------------

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "#e8a33d";
});

dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "";
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "";
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    dropzoneInner.hidden = true;
    scanFrame.hidden = false;
    previewImg.src = reader.result;
    scanLine.hidden = false;
  };
  reader.readAsDataURL(file);

  uploadFile(file);
}

async function uploadFile(file) {
  documentReady = false;
  statusRow.hidden = false;
  statusText.textContent = "Processing document…";
  resultsEl.hidden = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    scanLine.hidden = true;

    if (!response.ok) {
      statusText.textContent = data.error || "Upload failed.";
      return;
    }

    statusText.textContent = `Done — ${data.indexed_chunks} chunk(s) indexed`;
    renderResults(data);
    documentReady = data.indexed_chunks > 0;

    addChatMessage(
      documentReady
        ? "Document indexed. Ask a question below."
        : "No text was extracted from this document, so there's nothing to index yet.",
      "system"
    );
  } catch (err) {
    scanLine.hidden = true;
    statusText.textContent = "Upload failed — check the server logs.";
  }
}

function renderResults(data) {
  resultsEl.hidden = false;

  objectChips.innerHTML = "";
  if (data.objects.length === 0) {
    objectChips.innerHTML = '<span class="chip">No objects detected</span>';
  } else {
    data.objects.forEach((obj) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${obj.class} · ${(obj.confidence * 100).toFixed(0)}%`;
      objectChips.appendChild(chip);
    });
  }

  printedCount.textContent = `(${data.printed_text.length})`;
  printedText.textContent = data.printed_text.length
    ? data.printed_text.map((item) => item.text).join("\n")
    : "No printed text detected.";

  handwrittenText.textContent = data.handwritten_text || "No handwritten text detected.";

  indexNote.textContent = `Indexed chunks: ${data.indexed_chunks}`;
}

// ---------------- Chat handling ----------------

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = chatInput.value.trim();
  if (!question) return;

  addChatMessage(question, "user");
  chatInput.value = "";
  chatSend.disabled = true;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    if (!response.ok) {
      addChatMessage(data.error || "Something went wrong.", "error");
    } else {
      addChatMessage(data.answer, "assistant");
    }
  } catch (err) {
    addChatMessage("Couldn't reach the server.", "error");
  } finally {
    chatSend.disabled = false;
  }
});

function addChatMessage(text, role) {
  const msg = document.createElement("div");
  msg.className = `chat-msg chat-msg-${role}`;
  msg.textContent = text;
  chatLog.appendChild(msg);
  chatLog.scrollTop = chatLog.scrollHeight;
}
