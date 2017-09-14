from script.salience import eval
import sys

if len(sys.argv) < 3:
    print("Usage; run_eval.py gold_path system_path")
    exit(1)

eval.evaluate_entity_salience(sys.argv[1], sys.argv[2])
