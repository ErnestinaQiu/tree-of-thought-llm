#coding=utf8, ErnestinaQiu
import os
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

class ChatCompletion:
    def __init__(self):
        self.model_path = {"Llama-2-7b": "meta-llama/Llama-2-7b-chat"}

    def initiate_model(self, model_name):
        model_path = self.model_path["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype="float16")
        return {"tokenizer": self.tokenizer, "model": self.model}
    
    def create(self, **kwargs):
        messages = kwargs["messages"]
        # temperature = kwargs["temperature"]
        max_tokens = kwargs["max_tokens"]
        n = kwargs["n"]
        _stop = kwargs["stop"]
        
        self.initiate_model(model_name=kwargs["model"])
        input_features = self.tokenizer(messages)
        outputs = self.model.generate(**input_features, max_new_tokens=518)
        

