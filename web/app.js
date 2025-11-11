const uploadForm = document.getElementById('upload-form');
const textEl = document.getElementById('text');
const fileEl = document.getElementById('file');
const uploadStatus = document.getElementById('upload-status');

const askBtn = document.getElementById('askBtn');
const questionEl = document.getElementById('question');
const answerEl = document.getElementById('answer');

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  uploadStatus.textContent = '';

  const form = new FormData();
  const text = textEl.value.trim();
  const file = fileEl.files && fileEl.files[0];

  if (!text && !file) {
    uploadStatus.textContent = 'You must enter text or select a .txt file';
    return;
  }

  if (text) form.append('text', text);
  if (file) form.append('file', file);

  try {
    uploadForm.querySelector('button').disabled = true;
    uploadStatus.textContent = 'Uploading...';
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(errText || 'Upload error');
    }
    uploadStatus.textContent = 'Saved!';
    textEl.value = '';
    fileEl.value = '';
  } catch (err) {
    uploadStatus.textContent = 'Error: ' + err.message;
  } finally {
    uploadForm.querySelector('button').disabled = false;
    setTimeout(() => (uploadStatus.textContent = ''), 2500);
  }
});

askBtn.addEventListener('click', () => ask());
questionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    ask();
  }
});

function ask() {
  const q = questionEl.value.trim();
  if (!q) return;
  answerEl.textContent = '';

  const es = new EventSource('/api/query?q=' + encodeURIComponent(q));

  es.onmessage = (ev) => {
    // Append tokens
    try {
      // server escapes newlines as \n in data
      const text = ev.data.replace(/\\n/g, '\n');
      answerEl.textContent += text;
    } catch (_) {}
  };

  es.addEventListener('done', () => {
    es.close();
  });

  es.addEventListener('error', (ev) => {
    console.error('SSE error', ev);
    es.close();
  });
}
