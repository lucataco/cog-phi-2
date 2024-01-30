# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
import time
import torch
from weights_downloader import WeightsDownloader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"
MODEL_CACHE = "model-cache"
MODEL_URL = "https://weights.replicate.delivery/default/microsoft/phi-2.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        WeightsDownloader.download_if_not_exists(MODEL_URL, MODEL_CACHE)

        print("Loading pipeline...")
        torch.set_default_device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_length: int = Input(
            description="Max length", ge=0, le=2048, default=200
        ),
    ) -> str:
        """Run a single prediction on the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=max_length)
        result = self.tokenizer.batch_decode(outputs)[0]

        return result
