// script.js — BreastAI frontend logic

const uploadZone  = document.getElementById('upload-zone');
const fileInput   = document.getElementById('file-input');
const browseBtn   = document.getElementById('browse-btn');
const previewRow  = document.getElementById('preview-row');
const previewImg  = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const analyzeBtn  = document.getElementById('analyze-btn');
const btnText     = document.getElementById('btn-text');
const spinner     = document.getElementById('spinner');
const resetBtn    = document.getElementById('reset-btn');
const resultsSection = document.getElementById('results-section');

let selectedFile = null;

// ── Drag & Drop ──────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

['dragenter','dragover'].forEach(ev => {
  uploadZone.addEventListener(ev, (e) => {
    e.preventDefault(); uploadZone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(ev => {
  uploadZone.addEventListener(ev, () => uploadZone.classList.remove('dragover'));
});
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Handle selected file ──────────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src  = e.target.result;
    previewName.textContent = `${file.name}  (${(file.size/1024).toFixed(1)} KB)`;
    previewRow.style.display = 'flex';
    resultsSection.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

// ── Reset ─────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewRow.style.display = 'none';
  resultsSection.style.display = 'none';
  previewImg.src = '';
});

// ── Analyze ───────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  setLoading(true);

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  analyzeBtn.disabled = on;
  btnText.textContent  = on ? 'Analyzing…' : 'Analyze Image';
  spinner.style.display = on ? 'block' : 'none';
}

// ── Render results ────────────────────────────────────────────────────
function renderResults(data) {
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // ── Classification ────────────────────────────────────────────────
  const chip    = document.getElementById('diagnosis-chip');
  const label   = data.class;
  chip.textContent = label.charAt(0).toUpperCase() + label.slice(1);
  chip.className   = `diagnosis-chip ${label.toLowerCase()}`;

  document.getElementById('conf-value').textContent = data.confidence.toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('conf-bar').style.width = data.confidence + '%';
  }, 100);

  // Probability chips
  const probRow = document.getElementById('prob-row');
  probRow.innerHTML = '';
  const COLORS = { benign: '#34D399', malignant: '#F87171', normal: '#60A5FA' };
  for (const [cls, pct] of Object.entries(data.probabilities)) {
    const chip = document.createElement('span');
    chip.className = 'prob-chip';
    chip.style.color = COLORS[cls] || '#fff';
    chip.textContent = `${cls}: ${pct.toFixed(1)}%`;
    probRow.appendChild(chip);
  }

  // ── Segmentation ──────────────────────────────────────────────────
  const segCard = document.getElementById('seg-card');
  if (data.segmentation) {
    segCard.style.display = 'block';
    document.getElementById('seg-img').src = 'data:image/png;base64,' + data.segmentation;
  }

  // ── XAI images ────────────────────────────────────────────────────
  const xaiMap = {
    gradcam:   data.gradcam,
    gradcampp: data.gradcampp,
    scorecam:  data.scorecam,
    lime:      data.lime,
    shap:      data.shap,
  };

  for (const [key, b64] of Object.entries(xaiMap)) {
    const loader = document.getElementById(`loader-${key}`);
    const img    = document.getElementById(`img-${key}`);
    if (b64) {
      img.src = 'data:image/png;base64,' + b64;
      img.style.display = 'block';
      if (loader) loader.style.display = 'none';
    } else {
      if (loader) {
        loader.innerHTML = '<p style="color:#F87171">Not available (model not trained yet)</p>';
      }
    }
  }
}

// ── XAI tab switching ─────────────────────────────────────────────────
document.querySelectorAll('.xai-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.xai-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.xai-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panelId = 'panel-' + tab.dataset.tab;
    document.getElementById(panelId).classList.add('active');
  });
});
