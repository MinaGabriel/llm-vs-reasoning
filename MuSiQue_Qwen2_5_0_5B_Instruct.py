import pickle
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, AutoTokenizer
from MuSiQue import MuSiQue

# 1. Load a tiny model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0").eval()

# 2. Get 10 random MuSiQue questions that the model failed to answer
data = MuSiQue(n=10, tok=tok, model=model)

# 3. Save the results to a pickle file
output_filename = "musique_failed_samples.pkl"
with open(output_filename, "wb") as f:
  pickle.dump(list(data), f)
print(f"Successfully saved {len(data)} samples to {output_filename}")

# 4. Print results table
t = PrettyTable(["#", "Question", "Gold Answer", "Model Wrong Answer"])
t.align = "l"
for i, ex in enumerate(data, 1):
  t.add_row([i, ex["question"], ex["answer"], ex["model_answer"]])
print(t)