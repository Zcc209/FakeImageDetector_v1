const tabUrl = document.getElementById('tab-url');
const tabImage = document.getElementById('tab-image');
const urlSection = document.getElementById('url-section');
const imageSection = document.getElementById('image-section');
const imageUrlInput = document.getElementById('imageUrl');
const urlNoteInput = document.getElementById('urlNote');
const imageFileInput = document.getElementById('imageFile');
const uploadArea = document.getElementById('uploadArea');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const fileInfo = document.getElementById('fileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const resultBox = document.getElementById('resultBox');
const resultTitle = document.getElementById('resultTitle');
const resultText = document.getElementById('resultText');

let activeMode = 'url';

function switchTab(mode) {
  activeMode = mode;
  const isUrl = mode === 'url';
  tabUrl.classList.toggle('active', isUrl);
  tabImage.classList.toggle('active', !isUrl);
  urlSection.classList.toggle('hidden', !isUrl);
  imageSection.classList.toggle('hidden', isUrl);
  hideResult();
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showResult(title, text, isError = false) {
  resultTitle.textContent = title;
  resultText.textContent = text;
  resultBox.classList.add('show');
  resultBox.classList.toggle('error', isError);
}

function hideResult() {
  resultBox.classList.remove('show', 'error');
}

function loadFilePreview(file) {
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    fileName.textContent = file.name;
    fileInfo.textContent = `格式：${file.type || '未知'} ｜ 大小：${formatSize(file.size)}`;
    previewBox.classList.add('show');
  };
  reader.readAsDataURL(file);
}

function resetAll() {
  imageUrlInput.value = '';
  urlNoteInput.value = '';
  imageFileInput.value = '';
  previewImg.src = '';
  previewBox.classList.remove('show');
  fileName.textContent = '未選擇檔案';
  fileInfo.textContent = '檔案資訊將顯示於此。';
  hideResult();
  switchTab('url');
}

function formatAnalysisResult(data) {
  const gateOk = data?.gate_details?.is_valid;
  const riskLevel = data?.content?.risk_level ?? 'unknown';
  const riskScore = data?.content?.risk_score;
  const faceCount = data?.vision?.scrfd_face_count;
  const truforScore = data?.vision?.trufor_score;
  const truforErr = data?.vision?.trufor_error;
  const tampered = data?.vision?.is_tampered;

  const lines = [];
  lines.push(`流程狀態：${data?.status ?? 'unknown'}`);
  if (typeof gateOk === 'boolean') lines.push(`品質檢查：${gateOk ? '通過' : '未通過'}`);
  if (typeof faceCount === 'number') lines.push(`SCRFD 人臉數：${faceCount}`);
  if (typeof truforScore === 'number') {
    lines.push(`TruFor 分數：${truforScore.toFixed(6)} (${tampered ? '疑似篡改' : '偏正常'})`);
  } else if (truforErr) {
    lines.push(`TruFor：失敗 (${truforErr})`);
  }
  if (typeof riskScore === 'number') lines.push(`風險分數：${riskScore}`);
  lines.push(`風險等級：${riskLevel}`);
  if (data?.content?.explanation) lines.push(`說明：${data.content.explanation}`);
  return lines.join('\n');
}

async function callAnalyzeApi() {
  const endpoint = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:8000/api/analyze'
    : '/api/analyze';

  if (activeMode === 'url') {
    const imageUrl = imageUrlInput.value.trim();
    if (!imageUrl) {
      showResult('缺少圖片網址', '請先輸入要分析的圖片 URL。', true);
      return;
    }
    try {
      new URL(imageUrl);
    } catch {
      showResult('網址格式錯誤', '請輸入有效的圖片連結。', true);
      return;
    }

    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: imageUrl }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data?.message || `HTTP ${resp.status}`);
    }
    showResult('分析完成', formatAnalysisResult(data));
    return;
  }

  const file = imageFileInput.files[0];
  if (!file) {
    showResult('尚未上傳圖片', '請先選擇或拖曳一張圖片。', true);
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  const resp = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.message || `HTTP ${resp.status}`);
  }
  showResult('分析完成', formatAnalysisResult(data));
}

tabUrl.addEventListener('click', () => switchTab('url'));
tabImage.addEventListener('click', () => switchTab('image'));

imageFileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  loadFilePreview(file);
  if (file) hideResult();
});

['dragenter', 'dragover'].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
  });
});

uploadArea.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files[0] && files[0].type.startsWith('image/')) {
    imageFileInput.files = files;
    loadFilePreview(files[0]);
    hideResult();
  } else {
    showResult('格式不支援', '請拖曳圖片檔案，例如 JPG、PNG 或 WEBP。', true);
  }
});

analyzeBtn.addEventListener('click', async () => {
  const originalText = analyzeBtn.textContent;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = '分析中...';
  showResult('分析中', '正在呼叫後端模型流程，請稍候...');
  try {
    await callAnalyzeApi();
  } catch (err) {
    showResult('分析失敗', err?.message || '後端連線或推論發生錯誤。', true);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = originalText;
  }
});

resetBtn.addEventListener('click', resetAll);
