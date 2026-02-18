# from ultralytics.models.sam import SAM3VideoPredictor
import os

# demo dir
demo_data_dir = "demo_data/bottle/rgb"
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(demo_data_dir)) for f in fn if f.endswith(".png") or f.endswith(".jpg")]
files.sort()

from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3/sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image(files[0])

# Query with multiple text prompts
results = predictor(text=["hand", "bottle", "round sticker"])

boxes_object = results[0].boxes
masks_object = results[0].masks
print([mask.shape for mask in masks_object])
exit()
boxes = boxes_object.data.cpu().numpy()
print(boxes.shape)

# results = predictor(text=["bottle"])
# print(results)

# Works with descriptive phrases
# results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
# results = predictor(text=["a person"])