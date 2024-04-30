from models import Model
import logging
import torch
from peft import PeftModel
import transformers
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

logger = logging.getLogger(__name__)


class OdiaGenBengaliLlama(Model):
    def __init__(self, model_name, device) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device

    def activate_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config, device_map="auto"
        )
        self.model.eval()
        if torch.__version__ >= "2":
            self.model = torch.compile(self.model)

        logger.info(f"Model: {self.model_name} is activated.")

    def __evaluate(
        self,
        prompt,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        sequence = generation_output.sequence[0]
        output = self.tokenizer.decode(sequence, skip_special_tokens=True)
        return output.split("### Response:")[1].strip()

    def create_response(self, model_message) -> dict:
        content = self.__evaluate(prompt=model_message)

        response = {
            "content": content
        }

        return response

    def calculate_cost(self, input_tokens, output_tokens):
        pass
