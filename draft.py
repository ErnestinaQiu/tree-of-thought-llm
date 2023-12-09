import os 
import numpy as np
import json

pd_out_sp = "E:/study/dl/tot/tree-of-thought-llm/outputs.txt"
f = open(pd_out_sp, 'r')
pd_out = f.readlines()
f.close()
print(len(pd_out[0]))

meta_out_sp = "E:/study/dl/llama2/logprobs.txt"
f = open(meta_out_sp, 'r')
meta_out = f.readlines()
f.close()
print(len(meta_out[0]))
