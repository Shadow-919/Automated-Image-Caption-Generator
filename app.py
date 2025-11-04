import os
import torch
from flask import Flask, request, jsonify, render_template, url_for
from transformers import BertTokenizer
from dotenv import load_dotenv
import gdown

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform

load_dotenv()
app = Flask(__name__, template_folder="templates", static_folder="static")

device = torch.device("cpu")

# Model paths
model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)
file_name = "Model_EfficientNetB5_Transformer.pt"
model_path = os.path.join(model_dir, file_name)

gdrive_id = os.getenv("GDRIVE_FILE_ID")
if not gdrive_id:
    raise ValueError("Missing env GDRIVE_FILE_ID")

# Download model
if not os.path.exists(model_path):
    print("⬇️ Downloading model...")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, model_path, quiet=False)
    print("✅ Downloaded")


# Model hyperparams
embedding_dim = 512
max_seq_len = 128
encoder_layers = 6
decoder_layers = 12
num_heads = 8
dropout = 0.1
beam = 2

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = ImageCaptionModel(
    embedding_dim,
    tokenizer.vocab_size,
    max_seq_len,
    encoder_layers,
    decoder_layers,
    num_heads,
    dropout
)

state = torch.load(model_path, map_location=device)
model.load_state_dict(state, strict=False)

# Convert to FP16 for Render
for p in model.parameters():
    p.data = p.data.half()

model.to(device).eval()

# warm up EfficientNet
try:
    dummy = torch.zeros((1, 3, 224, 224)).half().to(device)
    _ = model.encoder(dummy)
    print("🔥 FP16 warm-up done")
except:
    pass

# Upload folder
upload_dir = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_dir, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_dir

@app.route("/")
def home():
    return render_template("Landing_page.html")

@app.route("/app")
def ui():
    return render_template("Caption_page.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    image = request.files["image"]
    save_path = os.path.join(upload_dir, image.filename)
    image.save(save_path)

    cap = generate_caption(
        model,
        save_path,
        transform,
        tokenizer,
        max_seq_len,
        beam,
        device,
        False
    )

    return jsonify({
        "caption": cap,
        "image_url": url_for("static", filename=f"uploads/{image.filename}", _external=True)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
