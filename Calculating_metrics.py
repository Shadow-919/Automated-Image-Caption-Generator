import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# ============================ #
#      Evaluation Settings     #
# ============================ #

# Path to ground truth COCO annotations (update this path in README for usage clarity)
ANNOTATION_FILE = r"D:\Aditya docs\BE\Final Year Project\Datasets\coco2014\annotations\captions_val2014.json"

# Path to generated captions
PREDICTIONS_FILE = "generated_captions.json"

# ============================ #
#     Load Ground Truth Data   #
# ============================ #

print("Loading COCO annotations...")
coco = COCO(ANNOTATION_FILE)
gts_raw = defaultdict(list)

for ann in coco.dataset['annotations']:
    gts_raw[ann['image_id']].append(ann['caption'])  # Collect reference captions for each image_id

# ============================ #
#     Load Generated Captions  #
# ============================ #

print("Loading generated predictions...")
with open(PREDICTIONS_FILE, 'r') as f:
    predictions = json.load(f)

res_raw = {}
for item in predictions:
    img_id = item['image_id']
    caption = item['caption']
    if isinstance(caption, str):
        res_raw[img_id] = [caption]  # Ensure it's in list format

# ============================ #
#   Filter Valid Image IDs     #
# ============================ #

valid_ids = set(gts_raw.keys()) & set(res_raw.keys())
gts = {k: gts_raw[k] for k in valid_ids}
res = {k: res_raw[k] for k in valid_ids}

print(f"Evaluating {len(valid_ids)} matched image_ids...")

# ============================ #
#     Run Evaluation Metrics   #
# ============================ #

scorers = [
    ("Bleu_1-4", Bleu(4)),
    ("ROUGE_L", Rouge()),
    ("CIDEr", Cider())
]

final_scores = {}

for name, scorer in scorers:
    print(f"Calculating {name}...")
    score, _ = scorer.compute_score(gts, res)

    if isinstance(score, list):  # For BLEU (multiple n-gram scores)
        for i, s in enumerate(score):
            final_scores[f"Bleu_{i+1}"] = s
    else:
        final_scores[name] = score

# ============================ #
#         Print Results        #
# ============================ #

print("\n=== Final Evaluation Metrics ===")
for metric, value in final_scores.items():
    print(f"{metric}: {value:.4f}")
