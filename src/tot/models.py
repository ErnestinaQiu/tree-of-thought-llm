import os
import backoff 
from llama2 import ChatCompletion

completion_tokens = prompt_tokens = 0


# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    chater = kwargs["chater"]
    print(kwargs["messages"])
    return chater.create(messages=kwargs["messages"], temperature=kwargs["temperature"])


def gpt(prompt, model="llama2", temperature=0.6, max_tokens=1000, n=1, stop=None) -> list:
    # 1
    # messages = [[{"role": "system", "content": "Do not make up answer if you don't know."},
    #             {"role": "user", "content": prompt}]]
    messages = [[{"role": "user", "content": prompt}]]
    return chatgpt(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)


def chatgpt(messages, model="llama2", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    if model == "llama2":
        chater = ChatCompletion()
    
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        print("messages: {}".format(messages))
        res = completions_with_backoff(chater=chater, model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

    
def gpt_usage(backend="llama2"):
    global completion_tokens, prompt_tokens
    if backend == "llama2":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    # elif backend == "gpt-3.5-turbo":
    #     cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
