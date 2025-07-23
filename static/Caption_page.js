document.addEventListener('DOMContentLoaded', function() {
  const fileInput = document.getElementById('fileInput');
  const dropBox = document.getElementById('dropBox');
  const generateButton = document.getElementById('generateButton');
  const resultModal = document.getElementById('resultModal');
  const backButton = document.getElementById('backButton');
  const imagePreview = document.getElementById('imagePreview');
  const captionContainer = document.getElementById('captionContainer');
  const speakerIcon = document.getElementById('speakerIcon');
  let droppedFile = null;
  let preloadedDataUrl = ""; // Stores the image data URL
  let cameraCapturedBlob = null; // Stores blob from camera capture

  // File input change event
  fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      droppedFile = file;
      cameraCapturedBlob = null;
      const reader = new FileReader();
      reader.onload = function(e) {
        preloadedDataUrl = e.target.result;
      };
      reader.readAsDataURL(file);
      updateDropBoxText(file.name);
    }
  });

  // Drag and drop events on dropBox
  dropBox.addEventListener('dragenter', function(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.add('active');
  });
  dropBox.addEventListener('dragover', function(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.add('active');
  });
  dropBox.addEventListener('dragleave', function(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.remove('active');
  });
  dropBox.addEventListener('drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
    dropBox.classList.remove('active');
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        droppedFile = file;
        cameraCapturedBlob = null;
        fileInput.value = "";
        const reader = new FileReader();
        reader.onload = function(e) {
          preloadedDataUrl = e.target.result;
        };
        reader.readAsDataURL(file);
        updateDropBoxText(file.name);
      }
    }
  });

  // Update dropBox layout to show uploaded file info inside the box
  function updateDropBoxText(fileName) {
    // Update the uploadedInfo container
    const uploadedInfo = document.getElementById("uploadedInfo");
    const uploadedFilename = document.getElementById("uploadedFilename");
    uploadedFilename.textContent = fileName;
    // Ensure uploadedInfo is inside dropBox
    if (uploadedInfo.parentNode !== dropBox) {
      dropBox.appendChild(uploadedInfo);
    }
    uploadedInfo.style.display = "block";
  
    // Hide the original drop text and buttons
    const dropText = dropBox.querySelector(".drop-text");
    if (dropText) dropText.style.display = "none";
    const orText = dropBox.querySelector(".or-text");
    if (orText) orText.style.display = "none";
    const buttonGroup = dropBox.querySelector(".button-group");
    if (buttonGroup) buttonGroup.style.display = "none";
    const imageFormats = dropBox.querySelector(".image-formats");
    if (imageFormats) imageFormats.style.display = "none";

    // Attach remove event for the "Remove" link
    document.getElementById("removeFileLink").addEventListener("click", removeUploadedImage);
  
    // Add the "uploaded" class so the dashed border stays blue
    dropBox.classList.add('uploaded');
  }
  
  function removeUploadedImage() {
    droppedFile = null;
    cameraCapturedBlob = null;
    preloadedDataUrl = "";
    fileInput.value = "";
  
    // Hide the uploadedInfo element but keep it in the DOM for reuse
    const uploadedInfo = document.getElementById("uploadedInfo");
    if (uploadedInfo) {
      uploadedInfo.style.display = "none";
      const uploadedFilename = document.getElementById("uploadedFilename");
      if (uploadedFilename) uploadedFilename.textContent = "";
    }
  
    // Remove captured image preview if present
    const capturedImage = document.getElementById("capturedImagePreview");
    if (capturedImage) {
      capturedImage.remove();
    }
  
    // Restore the original drop box texts and button group (clear any inline style)
    const dropText = dropBox.querySelector(".drop-text");
    if (dropText) dropText.style.display = "";
    const orText = dropBox.querySelector(".or-text");
    if (orText) orText.style.display = "";
    const buttonGroup = dropBox.querySelector(".button-group");
    if (buttonGroup) buttonGroup.style.display = "";
    const imageFormats = dropBox.querySelector(".image-formats");
    if (imageFormats) imageFormats.style.display = "";
  
    // Remove the "uploaded" class so the dashed border goes back to red
    dropBox.classList.remove('uploaded');
  
    // Also ensure the "active" class is removed
    dropBox.classList.remove('active');
    const dropOverlay = document.getElementById("dropOverlay");
    if (dropOverlay) {
      dropOverlay.style.display = "";
    }
  }
  
  
  
  
  
  

  // Generate caption button click event
  generateButton.addEventListener('click', async function() {
    let file;
    if (droppedFile) {
      file = droppedFile;
    } else if (cameraCapturedBlob) {
      file = new File([cameraCapturedBlob], "captured_image.png", { type: cameraCapturedBlob.type });
      cameraCapturedBlob = null;
    } else if (fileInput.files[0]) {
      file = fileInput.files[0];
    } else {
      alert('Please select an image first.');
      return;
    }
    resultModal.style.display = 'flex';
    if (preloadedDataUrl) {
      imagePreview.src = preloadedDataUrl;
      imagePreview.style.display = 'block';
    } else {
      imagePreview.style.display = 'none';
    }
    captionContainer.textContent = 'Generating caption...';
    captionContainer.style.fontFamily = "'Poppins'";
    const formData = new FormData();
    formData.append('image', file);
    try {
      const response = await fetch('/caption', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Server error');
      const data = await response.json();
      imagePreview.src = data.image_path + '?t=' + Date.now();
      captionContainer.textContent = data.caption;
    } catch (error) {
      console.error('Error:', error);
      captionContainer.textContent = 'Failed to generate caption.';
      captionContainer.style.fontFamily = "'Poppins'";
    }
  });

  // Back button event to close the modal
  backButton.addEventListener('click', function(e) {
    e.preventDefault();
    resultModal.style.display = 'none';
  });

  // Text-to-speech for caption (speaker icon)
  if (speakerIcon) {
    speakerIcon.addEventListener('click', function() {
      const text = captionContainer.textContent.trim();
      if (text.length > 0) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        speechSynthesis.speak(utterance);
      }
    });
  }

  /* --- Camera Functionality --- */
  let cameraStream;
  const openCameraButton = document.getElementById('openCameraButton');
  const cameraPreviewOverlay = document.getElementById('cameraPreviewOverlay');
  const cameraVideo = document.getElementById('cameraVideo');
  const captureButton = document.getElementById('captureButton');

  function attachCameraEventListeners() {
    const openBtn = document.getElementById('openCameraButton');
    const previewOverlay = document.getElementById('cameraPreviewOverlay');
    const videoElem = document.getElementById('cameraVideo');
    const capBtn = document.getElementById('captureButton');
    if (openBtn) {
      openBtn.addEventListener('click', async function(){
        previewOverlay.style.display = 'block';
        try {
          cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoElem.srcObject = cameraStream;
        } catch (err) {
          console.error("Error accessing camera: ", err);
          alert("Unable to access camera.");
          previewOverlay.style.display = 'none';
        }
      });
    }
    if (capBtn) {
      capBtn.addEventListener('click', function(){
        if (videoElem.videoWidth && videoElem.videoHeight) {
          const canvas = document.createElement('canvas');
          canvas.width = videoElem.videoWidth;
          canvas.height = videoElem.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(function(blob) {
            if (cameraStream) {
              cameraStream.getTracks().forEach(track => track.stop());
            }
            previewOverlay.style.display = 'none';
            const reader = new FileReader();
            reader.onload = function(e) {
              preloadedDataUrl = e.target.result;
              const uploadedInfo = document.getElementById("uploadedInfo");
              // Ensure uploadedInfo is inside dropBox
              if (uploadedInfo.parentNode !== dropBox) {
                dropBox.appendChild(uploadedInfo);
              }
              // Update the text to indicate a captured image
              const uploadedFilename = document.getElementById("uploadedFilename");
              uploadedFilename.textContent = "Captured Image";
              
              // Insert the captured image before the remove link inside uploadedInfo
              let capturedImage = document.getElementById("capturedImagePreview");
              if (!capturedImage) {
                capturedImage = document.createElement("img");
                capturedImage.id = "capturedImagePreview";
                capturedImage.style.maxWidth = "100%";
                capturedImage.style.maxHeight = "100%";
                const removeLink = uploadedInfo.querySelector(".remove-text");
                uploadedInfo.insertBefore(capturedImage, removeLink);
              }
              capturedImage.src = preloadedDataUrl;
              uploadedInfo.style.display = "block";
              
              // Hide the original dropBox text and button group
              const dropText = dropBox.querySelector(".drop-text");
              if (dropText) dropText.style.display = "none";
              const orText = dropBox.querySelector(".or-text");
              if (orText) orText.style.display = "none";
              const buttonGroup = dropBox.querySelector(".button-group");
              if (buttonGroup) buttonGroup.style.display = "none";
              const imageFormats = dropBox.querySelector(".image-formats");
              if (imageFormats) imageFormats.style.display = "none";
              
              // Attach the remove event to the remove link
              document.getElementById("removeFileLink").addEventListener("click", removeUploadedImage);
            
              // Add the "uploaded" class so the dashed border stays blue
              dropBox.classList.add('uploaded');
            };
            
            
            reader.readAsDataURL(blob);
            cameraCapturedBlob = blob;
          }, 'image/png');
        }
      });
    }
  }
  
  attachCameraEventListeners();
});
