import os
import torch
from flask import Flask, request, jsonify, render_template, url_for
from transformers import BertTokenizer

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Device setup
device = torch.device("cpu")  # Change to 'cuda' if GPU is desired

# Model hyperparameters (ensure consistency with training and caption scripts)
embedding_dim = 512
max_seq_len = 128
encoder_layers = 6
decoder_layers = 12
num_heads = 8
dropout = 0.1
beam_size = 2

# Pretrained model path (relative to repo root)
model_path = "./model/Model_EfficientNetB5_Transformer.pt"

# Load tokenizer and configure model
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

# Load model
model = ImageCaptionModel(**model_configs)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Warm-up model with dummy input to avoid delay on first request
try:
    dummy_input = torch.zeros((1, 3, 224, 224)).to(device)
    _ = model.encoder(dummy_input)
    print("Model warm-up completed.")
except Exception as e:
    print("Model warm-up failed:", e)

# Create uploads folder inside /static if not exists
upload_folder = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_folder, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_folder

# Route: Landing page (e.g., intro or splash screen)
@app.route("/")
def landing_page():
    return render_template("Landing_page.html")

# Route: Main captioning UI
@app.route("/app")
def app_page():
    return render_template("Caption_page.html")

# Route: Handle uploaded image and return caption
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

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
