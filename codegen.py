import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Codegen:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
        # model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
        self.model     = self.model.to(self.device)

    async def get_suggestion(self, input_str: str) -> str:
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.device)
        sample = self.model.generate(**inputs, max_new_tokens=32)
        output = self.tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        output_filtered = output.replace(input_str, "")
        output_arr =  output_filtered.split("\n")
        if len(output_arr) >= 2:
            return f"{output_arr[0]}\n{output_arr[1]}"
        else:
            return output_arr[0]
