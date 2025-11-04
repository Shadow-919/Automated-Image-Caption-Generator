document.addEventListener('DOMContentLoaded', function() {
  // Elements
  const fileInput = document.getElementById('fileInput');
  const dropBox = document.getElementById('dropBox');
  const uploadContent = document.getElementById('uploadContent');
  const filePreview = document.getElementById('filePreview');
  const previewImage = document.getElementById('previewImage');
  const previewFilename = document.getElementById('previewFilename');
  const removeFile = document.getElementById('removeFile');
  const generateButton = document.getElementById('generateButton');
  const resultModal = document.getElementById('resultModal');
  const closeModal = document.getElementById('closeModal');
  const resultOverlay = document.getElementById('resultOverlay');
  const resultImage = document.getElementById('resultImage');
  const captionText = document.getElementById('captionText');
  const speakButton = document.getElementById('speakButton');
  const copyButton = document.getElementById('copyButton');
  const dropOverlay = document.getElementById('dropOverlay');
  
  // Camera elements
  const openCameraButton = document.getElementById('openCameraButton');
  const cameraOverlay = document.getElementById('cameraOverlay');
  const cameraVideo = document.getElementById('cameraVideo');
  const captureButton = document.getElementById('captureButton');
  const closeCameraButton = document.getElementById('closeCameraButton');
  const switchCameraButton = document.getElementById('switchCameraButton');
  
  let selectedFile = null;
  let preloadedDataUrl = '';
  let cameraStream = null;
  let currentFacingMode = 'environment'; // Start with back camera

  // File input change
  fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file);
    }
  });

  // Drag and drop
  dropBox.addEventListener('dragenter', handleDragEnter);
  dropBox.addEventListener('dragover', handleDragOver);
  dropBox.addEventListener('dragleave', handleDragLeave);
  dropBox.addEventListener('drop', handleDrop);

  function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.add('drag-over');
  }

  function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    // Only remove class if leaving the dropBox itself, not child elements
    if (e.target === dropBox) {
      dropBox.classList.remove('drag-over');
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        handleFileSelect(file);
      }
    }
  }

  // Handle file selection
  function handleFileSelect(file) {
    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = function(e) {
      preloadedDataUrl = e.target.result;
      previewImage.src = preloadedDataUrl;
      previewFilename.textContent = file.name;
      
      uploadContent.style.display = 'none';
      filePreview.style.display = 'block';
      dropBox.classList.add('has-file');
    };
    reader.readAsDataURL(file);
  }

  // Remove file
  removeFile.addEventListener('click', function() {
    selectedFile = null;
    preloadedDataUrl = '';
    fileInput.value = '';
    previewImage.src = '';
    previewFilename.textContent = '';
    
    uploadContent.style.display = 'block';
    filePreview.style.display = 'none';
    dropBox.classList.remove('has-file');
  });

  // Camera functionality
  openCameraButton.addEventListener('click', async function() {
    try {
      // Try to open camera with environment (back) camera first
      cameraStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: currentFacingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      });
      cameraVideo.srcObject = cameraStream;
      cameraOverlay.style.display = 'block';
    } catch (err) {
      console.error('Error accessing camera:', err);
      // If back camera fails, try front camera
      try {
        currentFacingMode = 'user';
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: 'user',
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          } 
        });
        cameraVideo.srcObject = cameraStream;
        cameraOverlay.style.display = 'block';
      } catch (err2) {
        console.error('Error accessing front camera:', err2);
        alert('Unable to access camera. Please check permissions and try again.');
      }
    }
  });

  // Switch camera (front/back)
  switchCameraButton.addEventListener('click', async function() {
    if (cameraStream) {
      // Stop current stream
      cameraStream.getTracks().forEach(track => track.stop());
      
      // Toggle facing mode
      currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
      
      try {
        // Start new stream with opposite camera
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: currentFacingMode,
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          } 
        });
        cameraVideo.srcObject = cameraStream;
      } catch (err) {
        console.error('Error switching camera:', err);
        alert('Unable to switch camera. Your device may only have one camera.');
        // Try to restart with original camera
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        try {
          cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              facingMode: currentFacingMode,
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            } 
          });
          cameraVideo.srcObject = cameraStream;
        } catch (err2) {
          console.error('Error restarting camera:', err2);
          stopCamera();
        }
      }
    }
  });

  closeCameraButton.addEventListener('click', function() {
    stopCamera();
  });

  captureButton.addEventListener('click', function() {
    if (cameraVideo.videoWidth && cameraVideo.videoHeight) {
      const canvas = document.createElement('canvas');
      canvas.width = cameraVideo.videoWidth;
      canvas.height = cameraVideo.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
      
      canvas.toBlob(function(blob) {
        const file = new File([blob], 'captured_image.png', { type: 'image/png' });
        handleFileSelect(file);
        stopCamera();
      }, 'image/png');
    }
  });

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
    cameraOverlay.style.display = 'none';
    currentFacingMode = 'environment'; // Reset to back camera
  }

  // Generate caption
  generateButton.addEventListener('click', async function() {
    if (!selectedFile) {
      alert('Please select an image first.');
      return;
    }

    // Show modal
    resultModal.style.display = 'flex';
    resultImage.src = preloadedDataUrl;
    captionText.textContent = 'Generating your caption...';
    // captionText.style.background = 'linear-gradient(135deg, var(--text-gray), var(--text-gray))';
    // captionText.style.webkitBackgroundClip = 'text';
    // captionText.style.webkitTextFillColor = 'transparent';

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/caption', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Server error');

      const data = await response.json();
      resultImage.src = data.image_path + '?t=' + Date.now();
      captionText.textContent = data.caption;
      // captionText.style.background = 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))';
      // captionText.style.webkitBackgroundClip = 'text';
      // captionText.style.webkitTextFillColor = 'transparent';
    } catch (error) {
      console.error('Error:', error);
      captionText.textContent = 'Failed to generate caption. Please try again.';
      // captionText.style.background = '#ef4444';
      // captionText.style.webkitBackgroundClip = 'text';
      // captionText.style.webkitTextFillColor = 'transparent';
    }
  });

  // Close modal
  closeModal.addEventListener('click', closeResultModal);
  resultOverlay.addEventListener('click', closeResultModal);

  function closeResultModal() {
    resultModal.style.display = 'none';
  }

  // Text-to-speech
  speakButton.addEventListener('click', function() {
    const text = captionText.textContent.trim();
    if (text && text !== 'Generating your caption...' && text !== 'Failed to generate caption. Please try again.') {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 0.9;
      speechSynthesis.speak(utterance);
    }
  });

  // Copy caption
  copyButton.addEventListener('click', async function() {
    const text = captionText.textContent.trim();
    if (text && text !== 'Generating your caption...' && text !== 'Failed to generate caption. Please try again.') {
      try {
        await navigator.clipboard.writeText(text);
        const originalText = copyButton.innerHTML;
        copyButton.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
        copyButton.style.borderColor = 'var(--success-color)';
        copyButton.style.color = 'var(--success-color)';
        
        setTimeout(() => {
          copyButton.innerHTML = originalText;
          copyButton.style.borderColor = '';
          copyButton.style.color = '';
        }, 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  });

  // Prevent default drag behavior on document
  document.addEventListener('dragover', function(e) {
    e.preventDefault();
  });

  document.addEventListener('drop', function(e) {
    e.preventDefault();
  });
});