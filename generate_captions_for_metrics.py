import os
import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from models import ImageCaptionModel
from evaluation import generate_caption
from utils import transform
from transformers import BertTokenizer

# ======== CONFIGURATION ========
COCO_ROOT = r"D:\Aditya docs\BE\Final Year Project\Datasets\coco2014"
VAL_IMAGES_DIR = os.path.join(COCO_ROOT, "images", "val2014")
ANNOTATION_FILE = os.path.join(COCO_ROOT, "annotations", "captions_val2014.json")
MODEL_PATH = "D:\Aditya docs\BE\Final Year Project\Final Project (Image Caption Generator)\model\Model_EfficientNetB5_Transformer.pt"
BEAM_SIZE = 2
MAX_SEQ_LEN = 128
USE_GPU = torch.cuda.is_available()

# ======== DEVICE ========
device = torch.device("cuda" if USE_GPU else "cpu")
print(f"Using device: {device}")

# ======== LOAD TOKENIZER & MODEL ========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model_configs = {
    "embedding_dim": 512,
    "vocab_size": tokenizer.vocab_size,
    "max_seq_len": MAX_SEQ_LEN,
    "encoder_layers": 6,
    "decoder_layers": 12,
    "num_heads": 8,
    "dropout": 0.1
}

model = ImageCaptionModel(**model_configs)

state = torch.load(model_path, map_location=device, mmap=True)
model.load_state_dict(state, strict=False)

# Reduce memory: FP16
model.half()

model.to(device)
model.eval()

# Ensure input stays FP32 for image, convert to FP16 inside model


# ======== LOAD COCO DATASET ========
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()  # All 5000 validation images

results = []

# ======== GENERATE CAPTIONS ========
for img_id in tqdm(image_ids, desc="Generating captions"):
    img_info = coco.loadImgs(img_id)[0]
    image_path = os.path.join(VAL_IMAGES_DIR, img_info['file_name'])

    try:
        caption = generate_caption(
            model=model,
            image_path=image_path,
            transform=transform,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            beam_size=BEAM_SIZE,
            device=device,
            print_process=False
        )

        results.append({
            "image_id": img_id,
            "caption": caption
        })

    except Exception as e:
        print(f"[ERROR] {img_info['file_name']}: {e}")

# ======== SAVE GENERATED RESULTS ========
with open("generated_captions.json", "w") as f:
    json.dump(results, f)

# ======== EVALUATE WITH COCOEvalCap ========
print("Evaluating with COCOEvalCap...")
coco_res = coco.loadRes("generated_captions.json")
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.evaluate()
