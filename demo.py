import os
import json
import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task

args = argparse.Namespace(backend='spark-2.0', temperature=0.5, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

task = Game24Task()
ys, infos = solve(args, task, 900)
# print(ys[0])
# print(infos)

ys_path = os.path.join(os.getcwd(), "demo_ys.txt")
f = open(ys_path, 'w', encoding="utf8")
f.write(json.dumps(ys))
f.close()

infos_path = os.path.join(os.getcwd(), "demo_infos.txt")
f = open(infos_path, 'w', encoding='utf8')
f.write(json.dumps(infos))
f.close()