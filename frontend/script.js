/* ════════════════════════════════════════════
   StegoAI – script.js
   Pure Vanilla JS, no frameworks
   ════════════════════════════════════════════ */

const API = '';   // Same origin; change to 'http://localhost:5000' if serving separately

// ──────────────────────────────────────────────
//  Utilities
// ──────────────────────────────────────────────

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function showToast(msg, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const container = $('#toastContainer');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(20px)'; el.style.transition = '0.3s'; setTimeout(() => el.remove(), 350); }, duration);
}

function setLoading(show, text = 'Processing…') {
  const overlay = $('#overlay');
  $('#overlayText').textContent = text;
  overlay.style.display = show ? 'flex' : 'none';
}

function setBtnLoading(btn, loading) {
  const label = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.btn-spinner');
  btn.disabled = loading;
  label.hidden = loading;
  spinner.hidden = !loading;
}

function fmtDate(ts) {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString();
}

// ──────────────────────────────────────────────
//  Theme Toggle
// ──────────────────────────────────────────────

const themeToggle = $('#themeToggle');
const saved = localStorage.getItem('theme');
if (saved === 'light') document.body.classList.add('light');

themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('light');
  localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
});

// ──────────────────────────────────────────────
//  Tab Navigation
// ──────────────────────────────────────────────

$$('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    $$('.nav-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${tab}`).classList.add('active');
    if (tab === 'logs') loadLogs();
  });
});

// ──────────────────────────────────────────────
//  Drag & Drop + File Input Setup
// ──────────────────────────────────────────────

function setupDropzone(dropzoneId, inputId, previewId, previewWrapId, onFile) {
  const zone  = $(`#${dropzoneId}`);
  const input = $(`#${inputId}`);
  const preview = $(`#${previewId}`);
  const wrap    = $(`#${previewWrapId}`);

  zone.addEventListener('click', () => input.click());
  zone.querySelector('.drop-link')?.addEventListener('click', e => { e.stopPropagation(); input.click(); });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) { showToast('Please upload an image file.', 'error'); return; }
    if (file.size > 20 * 1024 * 1024) { showToast('File too large (max 20 MB).', 'error'); return; }
    const url = URL.createObjectURL(file);
    preview.src = url;
    zone.style.display = 'none';
    wrap.style.display = 'block';
    onFile(file);
  }

  return { reset() { zone.style.display = ''; wrap.style.display = 'none'; preview.src = ''; input.value = ''; } };
}

// ──────────────────────────────────────────────
//  ENCODE TAB
// ──────────────────────────────────────────────

let encodeFile = null;

const encodeDz = setupDropzone(
  'encodeDropzone', 'encodeImageInput', 'encodePreview', 'encodePreviewWrap',
  file => { encodeFile = file; }
);

$('#encodeClear').addEventListener('click', () => {
  encodeFile = null;
  encodeDz.reset();
});

// Char counter
const encodeMsg = $('#encodeMessage');
const encodeCharCount = $('#encodeCharCount');
encodeMsg.addEventListener('input', () => { encodeCharCount.textContent = encodeMsg.value.length; });

// Encode button
const encodeBtn = $('#encodeBtn');
encodeBtn.addEventListener('click', async () => {
  if (!encodeFile) { showToast('Please upload a cover image.', 'error'); return; }
  const message = encodeMsg.value.trim();
  if (!message) { showToast('Message cannot be empty.', 'error'); return; }

  setBtnLoading(encodeBtn, true);
  setLoading(true, 'Generating DESM & embedding…');

  try {
    const fd = new FormData();
    fd.append('image', encodeFile);
    fd.append('message', message);
    const pubKey = $('#encodePubKey').value.trim();
    if (pubKey) fd.append('public_key', pubKey);

    const res = await fetch(`${API}/encode`, { method: 'POST', body: fd });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Encoding failed.');

    showEncodeResult(data);
    showToast('Message embedded successfully!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setBtnLoading(encodeBtn, false);
    setLoading(false);
  }
});

function showEncodeResult(data) {
  const col = $('#encodeResultCol');
  col.style.display = 'block';
  col.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Images
  $('#resultOriginal').src = `data:image/png;base64,${data.original_b64}`;
  const stegoImg = $('#resultStego');
  stegoImg.src = `data:image/png;base64,${data.stego_image_b64}`;

  // Download link
  const dl = $('#downloadStego');
  dl.href = stegoImg.src;
  dl.download = `stego_${Date.now()}.png`;

  // Metrics
  const m = data.metrics;
  const mg = $('#metricsGrid');
  mg.innerHTML = '';
  const items = [
    { label: 'PSNR', value: `${m.psnr} dB`, base: m.baseline_psnr ? `Baseline: ${m.baseline_psnr} dB` : null },
    { label: 'SSIM', value: m.ssim, base: m.baseline_ssim ? `Baseline: ${m.baseline_ssim}` : null },
    { label: 'MSE',  value: m.mse,  base: m.baseline_mse  ? `Baseline: ${m.baseline_mse}`  : null },
    { label: 'Encode Time', value: `${data.encode_time}s` },
    { label: 'Message Size', value: `${data.message_length} chars` },
  ];
  items.forEach(i => {
    mg.innerHTML += `<div class="metric-card">
      <div class="metric-label">${i.label}</div>
      <div class="metric-value">${i.value}</div>
      ${i.base ? `<div class="metric-baseline">${i.base}</div>` : ''}
    </div>`;
  });

  // Integrity block
  const ib = $('#encodeIntegrityBlock');
  ib.innerHTML = `
    <div class="info-row"><span class="key">SHA-256 Hash</span><span class="val">${data.sha256?.substring(0,32)}…</span></div>
    <div class="info-row"><span class="key">Timestamp</span><span class="val">${fmtDate(data.timestamp)}</span></div>
  `;

  // Private key (if auto-generated)
  const pkBox = $('#privateKeyBox');
  if (data.private_key) {
    pkBox.style.display = 'block';
    $('#privateKeyText').textContent = data.private_key;
    $('#copyPrivKey').onclick = () => { navigator.clipboard.writeText(data.private_key); showToast('Private key copied!', 'success'); };
  } else {
    pkBox.style.display = 'none';
  }
}

// ──────────────────────────────────────────────
//  DECODE TAB
// ──────────────────────────────────────────────

let decodeFile = null;

const decodeDz = setupDropzone(
  'decodeDropzone', 'decodeImageInput', 'decodePreview', 'decodePreviewWrap',
  file => { decodeFile = file; }
);

const decodeBtn = $('#decodeBtn');
decodeBtn.addEventListener('click', async () => {
  if (!decodeFile) { showToast('Please upload a stego image.', 'error'); return; }
  const privKey = $('#decodePrivKey').value.trim();
  if (!privKey) { showToast('Private key is required to decode.', 'error'); return; }

  setBtnLoading(decodeBtn, true);
  setLoading(true, 'Extracting & decrypting…');

  try {
    const fd = new FormData();
    fd.append('image', decodeFile);
    fd.append('private_key', privKey);

    const res  = await fetch(`${API}/decode`, { method: 'POST', body: fd });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Decoding failed.');

    showDecodeResult(data);
    showToast(data.authentic ? 'Message decoded – Authentic!' : 'Decoded but integrity check FAILED!',
              data.authentic ? 'success' : 'error');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setBtnLoading(decodeBtn, false);
    setLoading(false);
  }
});

function showDecodeResult(data) {
  const panel = $('#decodeResult');
  const auth  = data.authentic;
  panel.style.display = 'block';
  panel.className = `result-panel ${auth ? 'result-authentic' : 'result-tampered'}`;
  panel.innerHTML = `
    <div class="result-inner">
      <div class="result-status">
        <div class="status-dot ${auth ? 'dot-green' : 'dot-red'}"></div>
        <span>${auth ? '✅ Authentic' : '⚠️ Tampered / Integrity Failed'}</span>
      </div>
      <div class="message-box">${escapeHtml(data.message)}</div>
      <div class="info-block" style="margin-top:1rem">
        <div class="info-row"><span class="key">Hash Match</span><span class="val">${data.hash_match ? '✅ Yes' : '❌ No'}</span></div>
        <div class="info-row"><span class="key">HMAC Match</span><span class="val">${data.hmac_match ? '✅ Yes' : '❌ No'}</span></div>
        <div class="info-row"><span class="key">Reason</span><span class="val">${data.reason}</span></div>
        <div class="info-row"><span class="key">Decode Time</span><span class="val">${data.decode_time}s</span></div>
        <div class="info-row"><span class="key">Embedded At</span><span class="val">${fmtDate(data.timestamp)}</span></div>
      </div>
    </div>`;
}

// ──────────────────────────────────────────────
//  VERIFY TAB
// ──────────────────────────────────────────────

let verifyFile = null;

setupDropzone(
  'verifyDropzone', 'verifyImageInput', 'verifyPreview', 'verifyPreviewWrap',
  file => { verifyFile = file; }
);

const verifyBtn = $('#verifyBtn');
verifyBtn.addEventListener('click', async () => {
  if (!verifyFile) { showToast('Please upload a stego image.', 'error'); return; }

  setBtnLoading(verifyBtn, true);
  setLoading(true, 'Verifying integrity…');
  try {
    const fd = new FormData();
    fd.append('image', verifyFile);
    // No private key needed — verification is purely hash/HMAC-based

    const res  = await fetch(`${API}/verify`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    const panel = $('#verifyResult');
    const auth  = data.authentic;
    panel.style.display = 'block';
    panel.className = `result-panel ${auth ? 'result-authentic' : 'result-tampered'}`;
    panel.innerHTML = `
      <div class="result-inner">
        <div class="result-status">
          <div class="status-dot ${auth ? 'dot-green' : 'dot-red'}"></div>
          <span>${auth ? '✅ Image is Authentic' : '⚠️ Tampering Detected'}</span>
        </div>
        <div class="info-block">
          <div class="info-row"><span class="key">Hash Match (SHA-256)</span><span class="val">${data.hash_match ? '✅ Yes' : '❌ No'}</span></div>
          <div class="info-row"><span class="key">HMAC Match</span><span class="val">${data.hmac_match ? '✅ Yes' : '❌ No'}</span></div>
          <div class="info-row"><span class="key">Reason</span><span class="val">${data.reason}</span></div>
          <div class="info-row"><span class="key">SHA-256 (stored)</span><span class="val">${data.sha256?.substring(0,32)}…</span></div>
          <div class="info-row"><span class="key">Embedded At</span><span class="val">${fmtDate(data.timestamp)}</span></div>
          <div class="info-row"><span class="key">Verify Time</span><span class="val">${data.verify_time}s</span></div>
        </div>
      </div>`;
    showToast(auth ? 'Image verified as authentic!' : 'Tampered image detected!', auth ? 'success' : 'error');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setBtnLoading(verifyBtn, false);
    setLoading(false);
  }
});

// ──────────────────────────────────────────────
//  KEYS TAB
// ──────────────────────────────────────────────

const genKeyBtn = $('#genKeyBtn');
genKeyBtn.addEventListener('click', async () => {
  setBtnLoading(genKeyBtn, true);
  setLoading(true, 'Generating RSA-2048 key pair…');
  try {
    const label = $('#keyLabel').value.trim() || `key_${Date.now()}`;
    const res  = await fetch(`${API}/generate-keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    $('#keyResult').style.display = 'block';
    $('#pubKeyOut').textContent  = data.public_key;
    $('#privKeyOut').textContent = data.private_key;
    $('#copyPub').onclick  = () => { navigator.clipboard.writeText(data.public_key);  showToast('Public key copied!',  'success'); };
    $('#copyPriv').onclick = () => { navigator.clipboard.writeText(data.private_key); showToast('Private key copied!', 'success'); };
    showToast('Key pair generated!', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setBtnLoading(genKeyBtn, false);
    setLoading(false);
  }
});

// ──────────────────────────────────────────────
//  LOGS TAB
// ──────────────────────────────────────────────

$('#refreshLogs').addEventListener('click', loadLogs);

async function loadLogs() {
  const el = $('#logsContent');
  el.innerHTML = '<p class="hint center">Loading…</p>';
  try {
    const res  = await fetch(`${API}/logs?n=30`);
    const data = await res.json();
    renderLogs(data);
  } catch {
    el.innerHTML = '<p class="hint center" style="color:var(--red)">Failed to load logs.</p>';
  }
}

function renderLogs(data) {
  const el = $('#logsContent');
  let html = '';

  if (data.encode_logs?.length) {
    html += '<p class="log-section-title">📤 Encode Operations</p><div class="log-table-wrap"><table>';
    html += '<tr><th>#</th><th>Image</th><th>Msg Len</th><th>PSNR</th><th>SSIM</th><th>Time</th><th>When</th></tr>';
    data.encode_logs.forEach(r => {
      html += `<tr><td>${r.id}</td><td>${r.image_name}</td><td>${r.message_len}</td>
               <td>${r.psnr}</td><td>${r.ssim}</td><td>${r.encode_time}s</td>
               <td>${fmtDate(r.timestamp)}</td></tr>`;
    });
    html += '</table></div>';
  }

  if (data.decode_logs?.length) {
    html += '<p class="log-section-title">📥 Decode Operations</p><div class="log-table-wrap"><table>';
    html += '<tr><th>#</th><th>Image</th><th>Authentic</th><th>Reason</th><th>Time</th><th>When</th></tr>';
    data.decode_logs.forEach(r => {
      const tag = r.authentic ? '<span class="tag-auth">✅ Yes</span>' : '<span class="tag-tampered">❌ No</span>';
      html += `<tr><td>${r.id}</td><td>${r.image_name}</td><td>${tag}</td>
               <td>${r.reason}</td><td>${r.decode_time}s</td>
               <td>${fmtDate(r.timestamp)}</td></tr>`;
    });
    html += '</table></div>';
  }

  if (!html) html = '<p class="hint center">No logs yet. Try encoding or decoding an image.</p>';
  el.innerHTML = html;
}

// ──────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
