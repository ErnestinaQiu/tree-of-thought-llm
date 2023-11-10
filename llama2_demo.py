from typing import List, Optional
import os
import json
import fire

from llama2.llama.llama import Llama, Dialog
from llama2 import ckpt_dir, tokenizer_path, max_seq_len, max_batch_size

messages = [
    [
     {"role": "system", "content": "Do not make up answer if you don't know."},
     {"role": "user", "content": "Who are you? Please introduce yourself."}
    ],
    [
        {"role": "system", "content": "Do not make up answer if you don't know."},
        {"role": "user", "content": "I want to start math learning, what tools should I prepare for that?"}
    ]
]

def main(
    messages: List[Dialog] = messages,
    ckpt_dir: str = ckpt_dir,
    tokenizer_path: str = tokenizer_path,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = max_seq_len,
    max_batch_size: int = max_batch_size,
    max_gen_len: Optional[int] = None):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        messages (list): There are two roles including "system" and "user". 
        --Example  [[{"role": "user", "content": "what is the recipe of mayonnaise?"}, {"role": "system", "content": "Always answer with Haiku"}]]
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    results = generator.chat_completion(
        messages,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    tmp_sp = os.path.join(os.getcwd(), "tmp/results.txt")
    print("results: \n", results)
    # f = open(tmp_sp, 'w', encoding="utf8")
    # f.wrire(str(results))
    # f.close()

    for dialog, result in zip(messages, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    # fire.Fire(main)
    main()


"""
test dialog: 
[
    [{"role": "user", "content": "Who are you? Please introduce yourself."},
     {"role": "system", "content": "Do not make up answer if you don't know."}
    ]
]

torchrun example_chat_completion.py --messages [
    [{"role": "user", "content": "Who are you? Please introduce yourself."},
     {"role": "system", "content": "Do not make up answer if you don't know."}
    ]
]
"""