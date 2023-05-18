import os
import datasets

COCO_DIR = os.path.join("/mnt/e/ML.Data/coco")
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", 
                           data_dir=COCO_DIR,
                           cache_dir='/mnt/e/hf_cache')

pass


from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")