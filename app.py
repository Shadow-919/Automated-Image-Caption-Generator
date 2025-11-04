import os
import torch
import gdown
from flask import Flask, request, jsonify, render_template, url_for
from transformers import BertTokenizer
from dotenv import load_dotenv  # ✅ For local .env loading

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform

# =========================================================
# 🌍 Load environment variables (works locally)
# =========================================================
load_dotenv()  # Will silently skip if no .env file exists

# =========================================================
# 🔧 Flask App Initialization
# =========================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

# =========================================================
# ⚙️ Device Setup
# =========================================================
device = torch.device("cpu")  # Render Free Tier = CPU only

# =========================================================
# 🧠 Model Configuration
# =========================================================
embedding_dim = 512
max_seq_len = 128
encoder_layers = 6
decoder_layers = 12
num_heads = 8
dropout = 0.1
beam_size = 2

# =========================================================
# 📁 Model Path + Secure Google Drive Download
# =========================================================
model_dir = "./model"
model_filename = "Model_EfficientNetB5_Transformer.pt"
model_path = os.path.join(model_dir, model_filename)

# Environment variable can be a Drive ID or full link
gdrive_value = os.getenv("GDRIVE_FILE_ID")

if not gdrive_value:
    raise ValueError("❌ GDRIVE_FILE_ID not found. Set it in your .env file or Render Environment tab!")

# Extract ID if a full URL is given
if "drive.google.com" in gdrive_value:
    # Handles both "file/d/ID/view" and "uc?id=ID" formats
    import re
    match = re.search(r'/d/([a-zA-Z0-9_-]+)|id=([a-zA-Z0-9_-]+)', gdrive_value)
    gdrive_file_id = match.group(1) or match.group(2)
else:
    gdrive_file_id = gdrive_value.strip()

gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

# =========================================================
# ⬇️ Download Model if Not Present
# =========================================================
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    print("⬇️ Model not found locally. Downloading from Google Drive...")
    try:
        gdown.download(gdrive_url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download model: {e}")
else:
    print("✅ Model already exists locally. Using cached version.")

# =========================================================
# 🧩 Load Tokenizer and Model
# =========================================================
print("⚙️ Loading tokenizer and model...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_configs = {
    "embedding_dim": embedding_dim,
    "vocab_size": tokenizer.vocab_size,
    "max_seq_len": max_seq_len,
    "encoder_layers": encoder_layers,
    "decoder_layers": decoder_layers,
    "num_heads": num_heads,
    "dropout": dropout,
}

model = ImageCaptionModel(**model_configs)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Warm-up to avoid first-request lag
try:
    dummy_input = torch.zeros((1, 3, 224, 224)).to(device)
    _ = model.encoder(dummy_input)
    print("🔥 Model warm-up completed.")
except Exception as e:
    print("⚠️ Model warm-up failed:", e)

# =========================================================
# 🗂️ Upload Folder Setup
# =========================================================
upload_folder = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_folder, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_folder

# =========================================================
# 🌐 Routes
# =========================================================
@app.route("/")
def landing_page():
    return render_template("Landing_page.html")

@app.route("/app")
def app_page():
    return render_template("Caption_page.html")

@app.route("/caption", methods=["POST"])
def caption_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    filename = image_file.filename
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    try:
        caption_text = generate_caption(
            model=model,
            image_path=image_path,
            transform=transform,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            beam_size=beam_size,
            device=device,
            print_process=False
        )
        return jsonify({
            "caption": caption_text,
            "image_path": url_for("static", filename=f"uploads/{filename}", _external=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# 🚀 Run Flask App (Render uses Gunicorn)
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
