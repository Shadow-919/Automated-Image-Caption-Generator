# 🖼️ Automated Image Caption Generator

**An end-to-end deep learning project that generates human-like captions for any given image using an Encoder-Decoder architecture and serves it via a web interface built with Flask.**

---

## 🧠 What is Image Captioning?

Image captioning is the task of generating a **natural language description** for an image. It combines the power of **Computer Vision** and **Natural Language Processing**, enabling machines to understand and describe visual content — just like a human.

This project implements a high-performance captioning model and deploys it as a fully functional **web app**.

---

## 🚀 Project Highlights

- 📸 Generates accurate, context-aware captions for any image
- ✨ Built using an **Encoder-Decoder** architecture  
- 🧪 Evaluated using BLEU (1–4), ROUGE, and CIDEr metrics
- 🔊 Integrated **Text-to-Speech (TTS)** to audibly read out the generated captions for enhanced accessibility.  
- 🌐 Model deployed on a responsive web interface via **Flask**

---

## 🏗️ Model Architecture

The image caption generator follows an **Encoder-Decoder** pipeline:

- **Encoder**:  
  Pretrained **EfficientNetB5** extracts dense feature vectors from input images.
  
- **Decoder**:  
  A **Transformer-based decoder** takes these features and generates captions word-by-word using learned language patterns.

This combination captures both **visual semantics** and **linguistic context**, making the captions more accurate and fluent.

---

## 📂 Dataset

The model was trained using the **MSCOCO 2014** dataset, a large-scale benchmark for image captioning and object detection tasks.

- **Dataset**: [Microsoft COCO 2014](https://cocodataset.org/#home)
- **Training Set**: ~82,783 images
- **Validation Set**: ~40,504 images
- **Captions**: Each image is annotated with 5 different human-written captions.
- **Purpose**: Used for training and evaluating the encoder-decoder model to generate high-quality image descriptions.
- **Preprocessing**: Includes image resizing, vocabulary creation, tokenization, and padding for sequence modeling.

---

## 📊 Evaluation Metrics

| Metric        | Score    |
|---------------|----------|
| BLEU-1        | 0.72     |
| BLEU-2        | 0.60     |
| BLEU-3        | 0.48     |
| BLEU-4        | 0.37     |
| ROUGE-L       | 0.66     |
| CIDEr         | 1.15     |

---

## 🌐 Project UI Screens

### 🖥️ Landing Page

<p align="center">
  <img src="assets/L1.PNG" width="80%" />
  <img src="assets/L2.PNG" width="80%" />
  <img src="assets/L3.PNG" width="80%" />
  <img src="assets/L4.PNG" width="80%" />
</p>

### 🧾 Caption Generation Page

<p align="center">
  <img src="assets/C1.PNG" width="80%" />
</p>

---

## 🖼️ Caption Outputs (Sample Results)

<table align="center">
  <tr>
    <td><img src="assets/O1.PNG" width="100%"></td>
    <td><img src="assets/O2.PNG" width="100%"></td>
  </tr>
  <tr>
    <td><img src="assets/O3.PNG" width="100%"></td>
    <td><img src="assets/O4.PNG" width="100%"></td>
  </tr>
</table>

---

## 🚫 Note

> This project is shared for **demonstration purposes only**.  
> Model files and setup instructions are intentionally excluded.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
