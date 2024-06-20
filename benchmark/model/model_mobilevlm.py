import os
import sys

import torch
from PIL import Image
from pathlib import Path

from benchmark.data.dataset import prepare_prompt

sys.path.append(str((Path(__file__).parent.parent.parent / "MobileVLM").resolve()))

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

class MobileVLM:
    def __init__(self, args):
        model = args.model
        self.model_name = model

        cache_dir = args.model_cache_dir

        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, cache_dir=cache_dir)
        tokenizer.add_tokens("<image>", special_tokens=True)
        self.model = model
        self.model = self.model.to("cuda")

        self.processor = tokenizer
        self.processor.padding_side = "left"
        self.image_processor = image_processor

        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task
        self.config = self.model.config

    def generate(self, batch):
        generation = self.model.generate(**batch, do_sample=False, max_new_tokens=10, min_new_tokens=1)
        generation[generation == -200] = 32000
        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        if "ASSISTANT:" in captions[0]:
            captions = [c.split("ASSISTANT:")[1].strip() for c in captions]
        if "[/INST]" in captions[0]:
            captions = [c.split("[/INST]")[1].strip() for c in captions]
        if "<|assistant|>" in captions[0]:
            captions = [c.split("<|assistant|>")[1].strip() for c in captions]
        if "Assistant:" in captions[0]:
            captions = [c.split("Assistant:")[1].strip() for c in captions]
        return captions


    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [Image.open(os.path.join(self.image_root, b["image"])).convert("RGB") for b in batch]

        images_tensor = process_images(images, self.image_processor, self.config).to(dtype=torch.float16)

        inputs = self.processor(prompts, return_tensors="pt", padding=True)
        inputs["input_ids"][inputs["input_ids"] == 32000] = -200
        inputs["images"] = images_tensor
        return inputs, labels, mapping, image_files
