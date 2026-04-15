let ws;
const videoFeed = document.getElementById('video-feed');
const setupMessage = document.getElementById('setup-message');
const evidenceFeed = document.getElementById('evidence-feed');
const indicator = document.getElementById('live-indicator');
const statusText = document.getElementById('status-text');
const judicialLogTerminal = document.getElementById('judicial-log-terminal');

let lastEvidenceTime = 0;
const DEDUPLICATION_WINDOW = 3.0; // Seconds

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function handleFileUpload(input) {
    if (!input.files || !input.files[0]) return;
    
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const startBtn = document.getElementById('start-btn');
    const urlInput = document.getElementById('video-input');
    
    startBtn.disabled = true;
    startBtn.innerText = "UPLOADING...";

    try {
        const response = await fetch('/upload_evidence', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.status === "SUCCESS") {
            urlInput.value = result.path;
            startDiscovery(); 
        } else {
            alert("Upload failed. Judicial handover error.");
            startBtn.disabled = false;
            startBtn.innerText = "START AUDIT";
        }
    } catch (err) {
        console.error("Upload Error:", err);
        startBtn.disabled = false;
        startBtn.innerText = "START AUDIT";
    }
}

async function startDiscovery() {
    const startBtn = document.getElementById('start-btn');
    const input = document.getElementById('video-input');
    const videoUrl = input.value.trim();
    if (!videoUrl) return;

    // --- BUTTON LOCK (Prevent Multiple Handshakes) ---
    startBtn.disabled = true;
    startBtn.innerText = "INITIALIZING...";
    
    // Cleanup previous state
    evidenceFeed.innerHTML = '';
    judicialLogTerminal.innerHTML = '<div class="log-item"><span class="log-tag">[SYSTEM]</span> Establishing secure handshake...</div>';
    setupMessage.style.display = 'none';
    videoFeed.style.display = 'block';

    const wsUrl = `ws://${window.location.hostname}:8989/ws/discover?video_path=${encodeURIComponent(videoUrl)}`;
    
    if (ws) ws.close();
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusText.innerText = "DISCOVERING";
        indicator.classList.add('active');
        startBtn.innerText = "ACTIVE SCAN";
        
        const logItem = document.createElement('div');
        logItem.className = 'log-item';
        logItem.innerHTML = `<span class="log-tag">[SYSTEM]</span> Judicial link established. Monitoring stream...`;
        judicialLogTerminal.appendChild(logItem);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.status === "COMPLETE") {
            statusText.innerText = "AUDIT COMPLETE";
            indicator.classList.remove('active');
            startBtn.disabled = false;
            startBtn.innerText = "START AUDIT";
            return;
        }

        if (data.status === "ERROR") {
            statusText.innerText = "HANDSHAKE ERROR";
            indicator.classList.remove('active');
            startBtn.disabled = false;
            startBtn.innerText = "START AUDIT";
            
            const logItem = document.createElement('div');
            logItem.className = 'log-item';
            logItem.innerHTML = `<span class="log-tag" style="color:var(--accent-red)">[JUDICIAL ERROR]</span> ${data.message}`;
            judicialLogTerminal.appendChild(logItem);
            return;
        }

        // --- NEW ASYNCHRONOUS INSIGHT HANDLER ---
        if (data.status === "INSIGHT") {
            // Update historical feed with a thumbnail if available and the AI legal brief
            addEvidenceCard(data); 
            updateJudicialLog(data);
            return;
        }

        // --- REAL-TIME DISCOVERY HANDLER ---
        if (data.status === "DISCOVERING") {
            // 1. Update Video Stream Immediately
            videoFeed.src = `data:image/jpeg;base64,${data.frame}`;

            // 2. Process Real-Time YOLO Detections - INSTANT FEEDBACK
            if (data.detections && data.detections.length > 0) {
                data.detections.forEach(det => {
                    // Create an instant card for every detection if it doesn't exist for this second
                    addEvidenceCard(det, data.timestamp);
                });
            }
        }
    };

    ws.onclose = () => {
        statusText.innerText = "OFFLINE";
        indicator.classList.remove('active');
        startBtn.disabled = false;
        startBtn.innerText = "START AUDIT";
        
        const logItem = document.createElement('div');
        logItem.className = 'log-item';
        logItem.innerHTML = `<span class="log-tag">[SYSTEM]</span> Discovery sequence terminated/offline.`;
        judicialLogTerminal.appendChild(logItem);
    };

    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
        const logItem = document.createElement('div');
        logItem.className = 'log-item';
        logItem.innerHTML = `<span class="log-tag" style="color:red">[ERROR]</span> Judicial link failure. Verify URL/Network.`;
        judicialLogTerminal.appendChild(logItem);
    };
}

function addEvidenceCard(data, timestamp = null) {
    const isInsight = data.status === "INSIGHT";
    const detections = isInsight ? data.detections : [data];
    const rawTs = isInsight ? data.timestamp : timestamp;
    const ts = parseFloat(rawTs);
    
    // Normalize to 0.5s grain for pure chronological pairing
    const timeIndex = Math.floor(ts * 2) / 2;
    const baseId = `card-group-${timeIndex.toFixed(1).replace('.', '-')}`;
    let card = document.getElementById(baseId);

    if (!card) {
        card = document.createElement('div');
        card.className = 'evidence-card';
        card.id = baseId;
        
        const labels = detections.map(d => d.label.toUpperCase()).sort().join(" + ");
        const timeLabel = formatTime(ts);
        const crop = isInsight ? data.thumbnail : (detections[0].crop || null);

        card.innerHTML = `
            <div class="card-image-wrap" style="background:#000; display:flex; align-items:center; justify-content:center; aspect-ratio:16/9;">
                ${crop ? `<img src="data:image/jpeg;base64,${crop}" id="${baseId}-img" style="width:100%; height:100%; object-fit:contain;" />` : '<div class="placeholder" style="color:#666; font-size:0.7rem;">CAPTURE PENDING...</div>'}
            </div>
            <div class="evidence-info">
                <div class="evidence-type">ENTRY_${timeLabel.replace(':', '')}</div>
                <div class="evidence-time">TAGS: ${labels}</div>
                <div class="evidence-brief" id="${baseId}-brief" style="margin-top:8px; font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">
                    ${isInsight ? data.legal_brief : 'ANALYZING EVIDENCE...'}
                </div>
            </div>
        `;

        // Insert at top
        evidenceFeed.insertBefore(card, evidenceFeed.firstChild);
    } else if (isInsight) {
        // Update existing card with Gemini Insight
        const briefEl = document.getElementById(`${baseId}-brief`);
        if (briefEl) {
            briefEl.innerHTML = data.legal_brief;
            briefEl.style.color = 'var(--text-primary)';
        }
        // Update thumbnail only if we don't have one or if Gemini provides a better one
        const imgEl = document.getElementById(`${baseId}-img`);
        if (imgEl && data.thumbnail) {
            imgEl.src = `data:image/jpeg;base64,${data.thumbnail}`;
        }
    }
}

function updateJudicialLog(data) {
    const logItem = document.createElement('div');
    logItem.className = 'log-item';
    
    const timestamp = formatTime(data.timestamp);
    const logTag = data.detections && data.detections.length > 0 ? "DISCOVERY" : "ANALYSIS";

    logItem.innerHTML = `
        <span class="log-time">[${timestamp}]</span>
        <span class="log-tag">#${logTag}</span>
        <span class="log-text">${data.legal_brief}</span>
    `;

    judicialLogTerminal.appendChild(logItem);
    
    // Auto-scroll to bottom of Reasoning Log (Right Column)
    judicialLogTerminal.scrollTop = judicialLogTerminal.scrollHeight;
}
