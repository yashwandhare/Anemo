// ==============================================================================
// ANEMO - Application JavaScript
// ==============================================================================
// Purpose: Handle user interactions, file uploads, and API communication
// Security: Client-side validation, file type/size checks, safe DOM manipulation
// ==============================================================================

// ==============================================================================
// Configuration & Constants
// ==============================================================================

const CONFIG = {
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
    ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp'],
    API_ENDPOINT: '/predict',
    API_TIMEOUT: 60000 // 60 second timeout
};

// ==============================================================================
// DOM Element References
// ==============================================================================

const els = {
    // Image Upload Controls
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    cameraInput: document.getElementById('cameraInput'),
    clinicalImage: document.getElementById('clinicalImage'),
    placeholder: document.getElementById('placeholder'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    // Result Display Elements
    riskScoreDisplay: document.getElementById('riskScoreDisplay'),
    riskFill: document.getElementById('riskFill'),
    riskMarker: document.getElementById('riskMarker'),
    diagnosisLabel: document.getElementById('diagnosisLabel'),
    confValue: document.getElementById('confValue'),
    logList: document.getElementById('logList'),
    
    // Heatmap Elements
    heatmapContainer: document.getElementById('heatmapContainer'),
    // Compatibility stub: older code paths referenced toggleHeatmap; keep harmless proxy
    toggleHeatmap: document.getElementById('toggleHeatmap') || { innerText: '', disabled: false, style: {} },

    // Log Modal Elements
    modal: document.getElementById('logModal'),
    closeModalBtn: document.getElementById('closeModalBtn'),
    modalImg: document.getElementById('modalImg'),
    modalClass: document.getElementById('modalClass'),
    modalConf: document.getElementById('modalConf'),
    modalRisk: document.getElementById('modalRisk'),
    modalNote: document.getElementById('modalNote'),

    // Camera Modal Elements
    cameraModal: document.getElementById('cameraModal'),
    closeCameraBtn: document.getElementById('closeCameraBtn'),
    cameraVideo: document.getElementById('cameraVideo'),
    cameraCanvas: document.getElementById('cameraCanvas'),
    captureBtn: document.getElementById('captureBtn'),
    retakeBtn: document.getElementById('retakeBtn'),
    useCaptureBtn: document.getElementById('useCaptureBtn')
};

// ==============================================================================
// State Management
// ==============================================================================

let currentFile = null;
let cameraStream = null;
let currentAnalysisData = null;  // EXPLAINABILITY: Store latest analysis for heatmap

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * Validate uploaded file meets security and format requirements
 * @param {File} file - The file to validate
 * @returns {Object} - {valid: boolean, error: string}
 */
function validateFile(file) {
    // Check if file exists
    if (!file) {
        return { valid: false, error: 'No file provided' };
    }

    // Check file size
    if (file.size === 0) {
        return { valid: false, error: 'File is empty' };
    }

    if (file.size > CONFIG.MAX_FILE_SIZE) {
        const maxSizeMB = CONFIG.MAX_FILE_SIZE / (1024 * 1024);
        return { valid: false, error: `File too large. Maximum size: ${maxSizeMB}MB` };
    }

    // Check file type
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        return { valid: false, error: 'Invalid file type. Please upload a JPG, PNG, or WebP image.' };
    }

    // Check file extension
    const fileName = file.name.toLowerCase();
    const hasValidExtension = CONFIG.ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
    
    if (!hasValidExtension) {
        return { valid: false, error: 'Invalid file extension' };
    }

    return { valid: true, error: null };
}

/**
 * Sanitize text content to prevent XSS
 * @param {string} text - Text to sanitize
 * @returns {string} - Sanitized text
 */
function sanitizeText(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Handle camera capture with device detection
 * Provides better UX for mobile and desktop devices
 */
function handleCameraCapture() {
    // Check if we're on a mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Check if MediaDevices API is available (for live camera access)
    const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    
    if (isMobile || !hasMediaDevices) {
        // Mobile or no MediaDevices support: use native camera file input
        // This provides the best experience on mobile devices
        els.cameraInput.click();
    } else {
        // Desktop with camera support: open live camera modal
        openLiveCamera();
    }
}

// ==============================================================================
// Live Camera Functions (Desktop)
// ==============================================================================

/**
 * Open live camera modal and start video stream
 */
async function openLiveCamera() {
    try {
        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user', // Front camera
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });

        // Set video source
        els.cameraVideo.srcObject = cameraStream;
        
        // Show modal
        els.cameraModal.classList.remove('hidden');
        
        // Reset UI state
        els.cameraVideo.classList.remove('hidden');
        els.cameraCanvas.classList.add('hidden');
        els.captureBtn.classList.remove('hidden');
        els.retakeBtn.classList.add('hidden');
        els.useCaptureBtn.classList.add('hidden');
        
    } catch (error) {
        console.error('Camera access error:', error);
        
        // Handle different error types
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            alert('Camera access denied. Please allow camera permissions and try again.');
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            alert('No camera found. Please use the upload button instead.');
        } else {
            alert('Unable to access camera. Please use the upload button instead.');
        }
        
        // Fallback to file input
        els.cameraInput.click();
    }
}

/**
 * Close camera modal and stop video stream
 */
function closeLiveCamera() {
    // Stop all video tracks
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    // Clear video source
    els.cameraVideo.srcObject = null;
    
    // Hide modal
    els.cameraModal.classList.add('hidden');
}

/**
 * Capture current video frame to canvas
 */
function capturePhoto() {
    const video = els.cameraVideo;
    const canvas = els.cameraCanvas;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas (flip horizontally for mirror effect)
    context.save();
    context.scale(-1, 1);
    context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    context.restore();
    
    // Hide video, show canvas
    els.cameraVideo.classList.add('hidden');
    els.cameraCanvas.classList.remove('hidden');
    
    // Update button visibility
    els.captureBtn.classList.add('hidden');
    els.retakeBtn.classList.remove('hidden');
    els.useCaptureBtn.classList.remove('hidden');
}

/**
 * Retake photo - show video stream again
 */
function retakePhoto() {
    els.cameraVideo.classList.remove('hidden');
    els.cameraCanvas.classList.add('hidden');
    els.captureBtn.classList.remove('hidden');
    els.retakeBtn.classList.add('hidden');
    els.useCaptureBtn.classList.add('hidden');
}

/**
 * Use captured photo - convert to file and load
 */
function useCapturedPhoto() {
    const canvas = els.cameraCanvas;
    
    // Convert canvas to blob
    canvas.toBlob((blob) => {
        if (!blob) {
            alert('Failed to capture image. Please try again.');
            return;
        }
        
        // Create a file from the blob
        const timestamp = new Date().getTime();
        const file = new File([blob], `camera-capture-${timestamp}.jpg`, {
            type: 'image/jpeg',
            lastModified: timestamp
        });
        
        // Close camera modal
        closeLiveCamera();
        
        // Load the captured file
        loadFile(file);
        
    }, 'image/jpeg', 0.92); // JPEG quality 92%
}

// ==============================================================================
// Report Download Functions
// ==============================================================================

/**
 * Trigger browser print dialog to save report as PDF
 * Print styles hide top bar, footer, logs, and show only content
 */
function downloadReport() {
    if (!currentAnalysisData) {
        alert('No analysis results to download. Please run an analysis first.');
        return;
    }
    
    // Trigger browser print dialog
    // CSS @media print rules handle hiding UI chrome
    window.print();
}

// ==============================================================================
// Event Listeners Setup
// ==============================================================================

// Upload button triggers file input
document.getElementById('uploadBtn').onclick = () => els.fileInput.click();

// Camera button triggers camera input with proper handling
document.getElementById('cameraBtn').onclick = () => handleCameraCapture();

// Log modal close button
els.closeModalBtn.onclick = () => els.modal.classList.add('hidden');

// Download button handler
const downloadBtn = document.getElementById('downloadBtn');
if (downloadBtn) {
    downloadBtn.onclick = () => downloadReport();
}

// Camera modal controls
els.closeCameraBtn.onclick = () => closeLiveCamera();
els.captureBtn.onclick = () => capturePhoto();
els.retakeBtn.onclick = () => retakePhoto();
els.useCaptureBtn.onclick = () => useCapturedPhoto();

// Close modals on background click
window.onclick = (e) => { 
    if (e.target === els.modal) {
        els.modal.classList.add('hidden');
    }
    if (e.target === els.cameraModal) {
        closeLiveCamera();
    }
};

// Close modals on Escape key
window.onkeydown = (e) => { 
    if (e.key === "Escape") {
        els.modal.classList.add('hidden');
        closeLiveCamera();
    }
};

// ============================================================================
// UI Behavior: Auto-hide Top Bar on Scroll
// ============================================================================
(function enableAutoHideTopBar(){
    const topBar = document.querySelector('.top-bar');
    if (!topBar) return;
    let lastY = window.scrollY;
    let ticking = false;

    const onScroll = () => {
        const currentY = window.scrollY;
        if (currentY > lastY && currentY > 64) {
            topBar.classList.add('is-hidden');
        } else {
            topBar.classList.remove('is-hidden');
        }
        lastY = currentY;
        ticking = false;
    };

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(onScroll);
            ticking = true;
        }
    }, { passive: true });
})();

// File input change handlers
[els.fileInput, els.cameraInput].forEach(input => {
    input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            loadFile(e.target.files[0]);
        }
    });
});

// Drag and drop handlers
els.dropZone.addEventListener('dragover', (e) => { 
    e.preventDefault(); 
    els.dropZone.style.borderColor = "var(--accent)";
});

els.dropZone.addEventListener('dragleave', () => { 
    els.dropZone.style.borderColor = "var(--border)";
});

els.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.dropZone.style.borderColor = "var(--border)";
    
    if (e.dataTransfer.files.length) {
        loadFile(e.dataTransfer.files[0]);
    }
});

// ==============================================================================
// File Handling Functions
// ==============================================================================

/**
 * Load and validate uploaded image file
 * @param {File} file - The image file to load
 */
function loadFile(file) {
    // Validate file before processing
    const validation = validateFile(file);
    
    if (!validation.valid) {
        alert(`Upload Error: ${validation.error}`);
        return;
    }

    currentFile = file;
    const reader = new FileReader();
    
    // Handle file read errors
    reader.onerror = () => {
        alert('Error reading file. Please try again.');
        currentFile = null;
    };
    
    // Process file once loaded
    reader.onload = (e) => {
        // Display the uploaded image
        els.clinicalImage.src = e.target.result;
        els.clinicalImage.classList.remove('hidden');
        els.placeholder.classList.add('hidden');
        
        // Enable analysis button
        els.analyzeBtn.disabled = false;
        els.analyzeBtn.innerText = "RUN ANALYSIS";
        
        // Reset all output displays
        resetOutputs();
    };
    
    reader.readAsDataURL(file);
}

/**
 * Reset all analysis result displays to default state
 */
function resetOutputs() {
    els.diagnosisLabel.innerText = "---";
    els.diagnosisLabel.className = "";
    els.riskScoreDisplay.innerText = "--";
    els.riskFill.style.width = "0%";
    els.riskMarker.style.left = "0%";
    els.confValue.innerText = "--%";
    els.heatmapContainer.style.display = 'flex';
    els.heatmapContainer.innerHTML = '<p class="heatmap-info">Upload an image to view the Grad-CAM overlay.</p>';
}

// ==============================================================================
// Analysis Functions
// ==============================================================================

/**
 * Main analysis handler - sends image to backend for processing
 */
els.analyzeBtn.onclick = async () => {
    // Safety check
    if (!currentFile) {
        alert('Please upload an image first');
        return;
    }
    
    // Update UI to processing state
    els.analyzeBtn.innerText = "PROCESSING...";
    els.analyzeBtn.disabled = true;

    // Prepare form data for upload
    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        // SECURITY: Add request timeout to prevent hanging
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);
        
        // Send request to backend API with Grad-CAM explainability enabled
        const res = await fetch(CONFIG.API_ENDPOINT + "?explain=true", { 
            method: "POST", 
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Handle HTTP errors
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${res.status}`);
        }
        
        // Parse response
        const data = await res.json();
        
        // Validate response data
        if (!data || typeof data.label === 'undefined' || typeof data.confidence === 'undefined') {
            throw new Error('Invalid response from server');
        }
        
        // Update UI with results
        displayResults(data);
        addLog(data);
        
    } catch (e) {
        if (e.name === 'AbortError') {
            console.error('Request timeout');
            alert('Analysis timeout. Please try again.');
        } else {
            console.error('Analysis error:', e);
            alert(`Analysis Error: ${e.message}`);
        }
        
        // Re-enable button for retry
        els.analyzeBtn.disabled = false;
        els.analyzeBtn.innerText = "RETRY ANALYSIS";
    }
};

// ==============================================================================
// Result Display Functions
// ==============================================================================

/**
 * Display analysis results in the UI
 * @param {Object} data - Analysis result data from API
 */
function displayResults(data) {
    // EXPLAINABILITY: Store data for heatmap display
    currentAnalysisData = data;
    
    // Debug: Log full response to check heatmap_url
    console.log('API Response:', data);
    console.log('Has heatmap_url:', 'heatmap_url' in data);
    
    // Update button state
    els.analyzeBtn.innerText = "ANALYSIS COMPLETE";
    els.analyzeBtn.disabled = false;

    // Update result image with cache-busting timestamp
    els.clinicalImage.src = data.boxed_image_url + "?t=" + Date.now();

    // Calculate risk score (store for logs)
    // If ANEMIC: risk = confidence, If NOT ANEMIC: risk = 1 - confidence
    const isAnemic = data.label === "ANEMIC";
    const risk = isAnemic ? (data.confidence / 100) : (1 - (data.confidence / 100));
    data.risk = risk;
    
    // Update risk meter display
    els.riskScoreDisplay.innerText = risk.toFixed(2);
    els.riskFill.style.width = `${risk * 100}%`;
    els.riskMarker.style.left = `${risk * 100}%`;

    // Update diagnosis label (sanitized)
    els.diagnosisLabel.innerText = sanitizeText(data.label);
    els.confValue.innerText = data.confidence + "%";

    // Apply color coding based on result
    if (isAnemic) {
        els.diagnosisLabel.className = "text-danger";
        els.riskFill.style.backgroundColor = "var(--danger)";
    } else {
        els.diagnosisLabel.className = "text-safe";
        els.riskFill.style.backgroundColor = "var(--accent)";
    }
    
    // EXPLAINABILITY: Handle heatmap if available
    if (data.heatmap_url) {
        console.log('Heatmap URL found, displaying...');
        displayHeatmap(data.heatmap_url);
    } else {
        console.warn('No heatmap_url in response');
        els.heatmapContainer.style.display = 'flex';
        els.heatmapContainer.innerHTML = '<p class="heatmap-info">Heatmap not available for this run.</p>';
    }
}

/**
 * EXPLAINABILITY: Display Grad-CAM heatmap
 * @param {string} heatmapUrl - URL to the heatmap image
 */
function displayHeatmap(heatmapUrl) {
    // Debug: log the heatmap URL
    console.log('Displaying heatmap URL:', heatmapUrl);
    
    // Clear previous content
    els.heatmapContainer.innerHTML = '';
    
    // Create image element
    const img = document.createElement('img');
    const cacheBustUrl = heatmapUrl + "?t=" + Date.now();
    img.src = cacheBustUrl;
    img.alt = 'Grad-CAM Heatmap';
    
    // Add error handler for image loading
    img.onerror = () => {
        console.error('Failed to load heatmap image:', cacheBustUrl);
        els.heatmapContainer.innerHTML = '<p style="color: var(--text-muted); padding: 20px;">Failed to load heatmap</p>';
    };
    
    // Add image to container
    els.heatmapContainer.appendChild(img);
    els.heatmapContainer.style.display = 'flex';
}

// ==============================================================================
// Log Management Functions
// ==============================================================================

/**
 * Add analysis result to session log history
 * @param {Object} data - Analysis result data to log
 */
function addLog(data) {
    // Create log item element
    const logItem = document.createElement('div');
    logItem.className = 'log-item';
    
    // Determine color class based on result
    const isAnemic = data.label === "ANEMIC";
    const colorClass = isAnemic ? 'text-danger' : 'text-safe';
    
    // Sanitize data before inserting
    const safeLabel = sanitizeText(data.label);
    const safeConfidence = parseFloat(data.confidence).toFixed(2);
    const safeRisk = typeof data.risk === 'number' ? data.risk.toFixed(2) : '--';
    const timestamp = new Date().toLocaleTimeString();

    // Build log item HTML (safe - using textContent below)
    logItem.innerHTML = `
        <div class="log-meta">
            <span>${timestamp}</span>
            <span>Conf: ${safeConfidence}%</span>
            <span>Risk: ${safeRisk}</span>
        </div>
        <div class="log-result ${colorClass}">
            ${safeLabel}
        </div>
    `;

    // Attach click handler to open modal with details
    logItem.onclick = () => openLogModal(data, colorClass);

    // Remove empty state message if present
    const empty = els.logList.querySelector('.empty-state');
    if (empty) {
        empty.remove();
    }
    
    // Add new log to top of list
    els.logList.prepend(logItem);
}

/**
 * Open modal with detailed log information
 * @param {Object} data - Log data to display
 * @param {string} colorClass - CSS class for result coloring
 */
function openLogModal(data, colorClass) {
    // Populate modal with sanitized data
    els.modalImg.src = data.boxed_image_url;
    els.modalClass.innerText = sanitizeText(data.label);
    els.modalClass.className = colorClass;
    els.modalConf.innerText = parseFloat(data.confidence).toFixed(2) + "%";
    els.modalRisk.innerText = typeof data.risk === 'number' ? parseFloat(data.risk).toFixed(2) : "--";
    els.modalNote.innerText = sanitizeText(data.note || "Analysis complete.");
    
    // Show modal
    els.modal.classList.remove('hidden');
}