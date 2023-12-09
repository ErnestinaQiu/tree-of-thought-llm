import os
import json
import time

st = time.time()
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat", dtype="float16")
input_features = tokenizer("please introduce yourself", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=518)
# outputs = model(**input_features) # output of logits layer of the model
# out_sp = os.path.join(os.getcwd(), "outputs.txt")
# f = open(out_sp, 'w', encoding="utf8")
# f.write(json.dumps(outputs[0].tolist()))
# f.close()

# print("model 2 \n", dir(model))
# print("len(outputs): {}".format(len(outputs)))
out_0 = tokenizer.batch_decode(outputs[0])
et = time.time()
print(out_0)
print("cost_time: {}".format(et-st))