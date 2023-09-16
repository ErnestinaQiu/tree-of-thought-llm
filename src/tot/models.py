import os
import backoff 
from spark.test import appid as spark_appid
from spark.test import api_secret as spark_api_secret
from spark.test import api_key as spark_api_key
from spark.test import domain as spark_domain
from spark.test import Spark_url
from spark.test import ChatCompletion as SparkChatCompletion

completion_tokens = prompt_tokens = 0

@backoff.on_exception(backoff.expo, max_tries=3)
def completions_with_backoff(**kwargs):
    return SparkChatCompletion.create(**kwargs)

def gpt(prompt, model="spark-2.0", temperature=0.5, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="spark-2.0", temperature=0.5, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)

        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="spark-2.0"):
    global completion_tokens, prompt_tokens
    if backend == "spark-2.0":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "spark-1.5":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
