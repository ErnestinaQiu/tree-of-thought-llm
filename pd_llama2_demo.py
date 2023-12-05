import time

st = time.time()
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat", dtype="float16")
input_features = tokenizer("please introduce yourself", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=1024)
print("len(outputs): {}".format(len(outputs)))
out_0 = tokenizer.batch_decode(outputs[0])
out_1 = tokenizer.batch_decode(outputs[1])
et = time.time()
print(out_0)
print("cost_time: {}".format(et-st))