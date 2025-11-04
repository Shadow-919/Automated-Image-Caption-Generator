import os
import torch
from flask import Flask, request, jsonify, render_template, url_for
from transformers import BertTokenizer
from dotenv import load_dotenv
import gdown

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform  # only transform now

# Load env for local dev
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Force CPU (Render free tier)
device = torch.device("cpu")

# Model directory & file
model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)
model_file = "Model_EfficientNetB5_Transformer.pt"
model_path = os.path.join(model_dir, model_file)

# Google Drive file ID
gdrive_file_id = os.getenv("GDRIVE_FILE_ID")
if not gdrive_file_id:
    raise ValueError("❌ GDRIVE_FILE_ID not set. Add it in Render Environment Variables.")

# Download model if missing
def download_model_if_needed():
    if os.path.exists(model_path):
        print("✅ Model already exists locally")
        return
    
    print("⬇️ Model not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, model_path, quiet=False)
    print("✅ Model downloaded successfully!")

download_model_if_needed()

# Model params
embedding_dim = 512
max_seq_len = 128
encoder_layers = 6
decoder_layers = 12
num_heads = 8
dropout = 0.1
beam_size = 2

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Build model
model = ImageCaptionModel(
    embedding_dim=embedding_dim,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=max_seq_len,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
    num_heads=num_heads,
    dropout=dropout,
)

# Load weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Warm-up once
try:
    dummy = torch.zeros((1, 3, 224, 224)).to(device)
    _ = model.encoder(dummy)
    print("🔥 Model warm-up complete")
except Exception as e:
    print("Warm-up skipped:", e)

# Upload folder
upload_dir = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_dir, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_dir

# Routes
@app.route("/")
def home():
    return render_template("Landing_page.html")

@app.route("/app")
def caption_ui():
    return render_template("Caption_page.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files["image"]
    filename = image.filename
    save_path = os.path.join(upload_dir, filename)
    image.save(save_path)

    try:
        caption_text = generate_caption(
            model=model,
            image_path=save_path,
            transform=transform,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            beam_size=beam_size,
            device=device,
            print_process=False
        )

        return jsonify({
            "caption": caption_text,
            "image_url": url_for("static", filename=f"uploads/{filename}", _external=True)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
