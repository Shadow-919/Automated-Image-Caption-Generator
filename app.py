import os
import torch
from flask import Flask, request, jsonify, render_template, url_for
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform

# Load .env for local dev; on HF Spaces, set secrets in the UI (HF_TOKEN)
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# Config via env
# -----------------------------
# Example: MODEL_REPO="your-username/your-private-model-repo"
MODEL_REPO = os.getenv("MODEL_REPO", "").strip()
MODEL_FILE = os.getenv("MODEL_FILE", "Model_EfficientNetB5_Transformer.pt").strip()
HF_TOKEN   = os.getenv("HF_TOKEN", "").strip()  # Set in HF Space secrets (Write token)

if not MODEL_REPO:
    raise RuntimeError("MODEL_REPO not set. Add it as an env/secret in your Space.")

# Force CPU on HF free tier
device = torch.device("cpu")

# Where to stage model in ephemeral storage
os.makedirs("/tmp/models", exist_ok=True)
local_model_path = os.path.join("/tmp/models", MODEL_FILE)

# -----------------------------
# Fetch model weights (private)
# -----------------------------
def ensure_model_downloaded():
    if os.path.exists(local_model_path):
        print(f"✅ Using cached model at {local_model_path}")
        return local_model_path

    print(f"⬇️ Downloading model '{MODEL_FILE}' from private repo '{MODEL_REPO}' ...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=HF_TOKEN or None,          # token required for private repos
        repo_type="model",               # model repo type
        local_dir="/tmp/models",         # ephemeral path
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded to {path}")
    return path

model_path = ensure_model_downloaded()

# -----------------------------
# Build and load model (full precision)
# -----------------------------
embedding_dim   = int(os.getenv("EMBED_DIM", "512"))
max_seq_len     = int(os.getenv("MAX_SEQ_LEN", "128"))
encoder_layers  = int(os.getenv("ENCODER_LAYERS", "6"))
decoder_layers  = int(os.getenv("DECODER_LAYERS", "12"))
num_heads       = int(os.getenv("NUM_HEADS", "8"))
dropout         = float(os.getenv("DROPOUT", "0.1"))
beam_size       = int(os.getenv("BEAM_SIZE", "2"))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = ImageCaptionModel(
    embedding_dim=embedding_dim,
    vocab_size=tokenizer.vocab_size,
    max_seq_len=max_seq_len,
    encoder_layers=encoder_layers,
    decoder_layers=decoder_layers,
    num_heads=num_heads,
    dropout=dropout,
)

# Load full-precision weights
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state, strict=False)
model.to(device).eval()

# Optional quick warmup
try:
    with torch.no_grad():
        _ = model.encoder(torch.zeros(1, 3, 224, 224))
    print("🔥 Warmup complete")
except Exception as e:
    print("Warmup skipped:", e)

# Upload dir for served images
upload_dir = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_dir, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_dir

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    # If you don’t use templates, return a simple HTML here or serve your existing pages.
    return render_template("Landing_page.html") if os.path.exists(os.path.join("templates","Landing_page.html")) \
        else "App is running. POST an image to /caption."

@app.route("/app")
def caption_ui():
    return render_template("Caption_page.html") if os.path.exists(os.path.join("templates","Caption_page.html")) \
        else "Upload form not bundled. Use API POST /caption."

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = image.filename
    save_path = os.path.join(upload_dir, filename)
    image.save(save_path)

    try:
        text = generate_caption(
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
            "caption": text,
            "image_url": url_for("static", filename=f"uploads/{filename}", _external=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Local dev entrypoint (HF uses gunicorn via Dockerfile)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
