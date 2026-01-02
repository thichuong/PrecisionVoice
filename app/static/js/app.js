/**
 * PrecisionVoice - Frontend Application Logic
 * Handles file upload, transcription requests, and result display.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const elements = {
        // Upload
        dropZone: document.getElementById('drop-zone'),
        fileInput: document.getElementById('file-input'),
        fileInfo: document.getElementById('file-info'),
        fileName: document.getElementById('file-name'),
        fileSize: document.getElementById('file-size'),
        clearBtn: document.getElementById('clear-btn'),
        transcribeBtn: document.getElementById('transcribe-btn'),

        // Sections
        uploadSection: document.getElementById('upload-section'),
        processingSection: document.getElementById('processing-section'),
        resultsSection: document.getElementById('results-section'),
        errorSection: document.getElementById('error-section'),

        // Processing
        processingStatus: document.getElementById('processing-status'),
        progressFill: document.getElementById('progress-fill'),

        // Results
        speakerCount: document.getElementById('speaker-count'),
        durationInfo: document.getElementById('duration-info'),
        processingTime: document.getElementById('processing-time'),
        transcriptContainer: document.getElementById('transcript-container'),
        downloadTxt: document.getElementById('download-txt'),
        downloadSrt: document.getElementById('download-srt'),
        newUploadBtn: document.getElementById('new-upload-btn'),

        // Error
        errorMessage: document.getElementById('error-message'),
        retryBtn: document.getElementById('retry-btn')
    };

    let selectedFile = null;

    // =====================
    // Event Listeners
    // =====================

    // Click to upload
    elements.dropZone.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag and drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('dragover');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('dragover');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Clear file
    elements.clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    // Transcribe button
    elements.transcribeBtn.addEventListener('click', () => {
        if (selectedFile) {
            startTranscription();
        }
    });

    // New upload button
    elements.newUploadBtn.addEventListener('click', resetToUpload);

    // Retry button
    elements.retryBtn.addEventListener('click', resetToUpload);

    // =====================
    // File Handling
    // =====================

    function handleFileSelection(file) {
        const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a',
            'audio/ogg', 'audio/flac', 'audio/webm', 'video/webm'];
        const allowedExtensions = ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm'];

        // Check file extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            showError(`Unsupported file type: .${ext}. Supported: ${allowedExtensions.join(', ')}`);
            return;
        }

        // Check file size (100MB limit)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(`File too large. Maximum size: 100MB`);
            return;
        }

        selectedFile = file;

        // Update UI
        elements.fileName.textContent = file.name;
        elements.fileSize.textContent = formatFileSize(file.size);
        elements.fileInfo.classList.remove('hidden');
        elements.transcribeBtn.disabled = false;

        // Hide drop zone text
        elements.dropZone.style.display = 'none';
    }

    function clearFileSelection() {
        selectedFile = null;
        elements.fileInput.value = '';
        elements.fileInfo.classList.add('hidden');
        elements.transcribeBtn.disabled = true;
        elements.dropZone.style.display = 'block';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // =====================
    // Transcription
    // =====================

    async function startTranscription() {
        if (!selectedFile) return;

        // Show processing UI
        showSection('processing');
        updateProgress(0, 'Preparing file...');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Processing failed');
            }

            // Read stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process SSE format: "data: {...}\n\n"
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep partial line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            handleProgressUpdate(data);
                        } catch (e) {
                            console.error('Failed to parse status update:', e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Processing error:', error);
            showError(error.message || 'An error occurred during processing');
        }
    }

    function handleProgressUpdate(data) {
        if (data.status === 'processing') {
            updateProgress(data.progress, data.message);
        } else if (data.status === 'completed') {
            updateProgress(100, 'Complete!');
            displayResults(data.result);
        } else if (data.status === 'error') {
            showError(data.message);
        }
    }

    function updateProgress(percent, status) {
        elements.progressFill.style.width = `${percent}%`;
        if (status) {
            elements.processingStatus.textContent = status;
        }
    }

    // =====================
    // Results Display
    // =====================

    function displayResults(result) {
        // Update metadata
        elements.speakerCount.textContent = `${result.num_speakers} speaker${result.num_speakers !== 1 ? 's' : ''}`;
        elements.durationInfo.textContent = formatDuration(result.duration);
        elements.processingTime.textContent = `${result.processing_time}s`;

        // Set download links
        elements.downloadTxt.href = result.download_txt;
        elements.downloadSrt.href = result.download_srt;

        // Render transcript segments
        renderTranscript(result.segments);

        // Show results section
        showSection('results');
    }

    function renderTranscript(segments) {
        elements.transcriptContainer.innerHTML = '';

        const speakerColors = {};
        let colorIndex = 0;

        segments.forEach((segment) => {
            // Assign color to speaker
            if (!(segment.speaker in speakerColors)) {
                colorIndex++;
                speakerColors[segment.speaker] = `speaker-${Math.min(colorIndex, 5)}`;
            }

            const segmentEl = document.createElement('div');
            segmentEl.className = `segment ${speakerColors[segment.speaker]}`;

            segmentEl.innerHTML = `
                <div class="segment-header">
                    <span class="segment-speaker">${escapeHtml(segment.speaker)}</span>
                    <span class="segment-time">${formatTime(segment.start)} - ${formatTime(segment.end)}</span>
                </div>
                <p class="segment-text">${escapeHtml(segment.text)}</p>
            `;

            elements.transcriptContainer.appendChild(segmentEl);
        });
    }

    function formatTime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);

        if (h > 0) {
            return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function formatDuration(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // =====================
    // UI State Management
    // =====================

    function showSection(section) {
        elements.uploadSection.classList.add('hidden');
        elements.processingSection.classList.add('hidden');
        elements.resultsSection.classList.add('hidden');
        elements.errorSection.classList.add('hidden');

        switch (section) {
            case 'upload':
                elements.uploadSection.classList.remove('hidden');
                break;
            case 'processing':
                elements.processingSection.classList.remove('hidden');
                break;
            case 'results':
                elements.resultsSection.classList.remove('hidden');
                break;
            case 'error':
                elements.errorSection.classList.remove('hidden');
                break;
        }
    }

    function showError(message) {
        elements.errorMessage.textContent = message;
        showSection('error');
    }

    function resetToUpload() {
        clearFileSelection();
        showSection('upload');
        updateProgress(0, 'Uploading file...');
    }
});
