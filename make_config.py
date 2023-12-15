import os
import yaml

def make_db_config():
    llm = {
            "llama-2-7b-chat":   {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-7b-chat/",
                                  "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            "llama-2-7b":   {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-7b/",
                             "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            "llama-2-13b-chat": {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-13b-chat/",
                                 "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            "llama-2-13b":  {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-13b/",
                             "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            "llama-2-70b-chat":  {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-70b/",
                                  "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            "llama-2-70b":  {"ckpt_dir": "/mnt/e/study/dl/llama2/llama-2-70b/",
                             "tokenizer_path": "/mnt/e/study/dl/llama2/tokenizer.model"},
            }
    llm_path = os.path.join(os.getcwd(), "llm_config.yml")
    with open(llm_path, 'w', encoding="utf8") as f:
        yaml.dump(llm, f)
    
    
if __name__ == "__main__":
    make_db_config()