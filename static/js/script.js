// ===== DOM Elements =====
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropContent = document.getElementById('dropContent');
const previewContainer = document. getElementById('previewContainer');
const preview = document.getElementById('preview');
const removeBtn = document.getElementById('removeBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const uploadForm = document.getElementById('uploadForm');

// ===== File Input Change =====
if (fileInput) {
    fileInput.addEventListener('change', handleFileSelect);
}

// ===== Drag & Drop Events =====
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList. remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: fileInput });
        }
    });

    // Click to upload
    dropZone.addEventListener('click', (e) => {
        if (e.target === dropZone || dropContent.contains(e.target)) {
            fileInput.click();
        }
    });
}

// ===== Handle File Selection =====
function handleFileSelect(e) {
    const file = e.target.files[0];
    
    if (file && file.type.startsWith('image/')) {
        // Show preview
        const reader = new FileReader();
        reader.onload = (event) => {
            preview.src = event.target.result;
            dropContent.style.display = 'none';
            previewContainer. classList.add('active');
            
            // Update file info
            fileName.textContent = file.name;
            
            // Enable predict button
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

// ===== Remove Image =====
if (removeBtn) {
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        
        // Reset everything
        fileInput.value = '';
        preview.src = '';
        previewContainer.classList.remove('active');
        dropContent.style. display = 'block';
        fileName.textContent = 'No file selected';
        predictBtn.disabled = true;
    });
}

// ===== Form Submit - Show Loading =====
if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
        const btnText = predictBtn.querySelector('. btn-text');
        const btnLoader = predictBtn.querySelector('.btn-loader');
        
        if (btnText && btnLoader) {
            btnText.style.display = 'none';
            btnLoader.style.display = 'inline-block';
        }
        
        predictBtn.disabled = true;
    });
}

// ===== Animate Confidence Bar on Result Page =====
document.addEventListener('DOMContentLoaded', () => {
    const confidenceBar = document.querySelector('.confidence-bar');
    if (confidenceBar) {
        // Trigger animation
        const width = confidenceBar.style.width;
        confidenceBar.style.width = '0';
        setTimeout(() => {
            confidenceBar.style.width = width;
        }, 100);
    }
});