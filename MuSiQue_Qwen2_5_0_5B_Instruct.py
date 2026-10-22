from transformers import AutoTokenizer, AutoModelForCausalLM
from prettytable import PrettyTable
from MuSiQue import MuSiQue

# 1. Load a tiny model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0").eval()

# 2. Get 10 random MuSiQue questions that the model failed to answer
data = MuSiQue(n=10, tok=tok, model=model)

# 3. Print results
t = PrettyTable(["#", "Question", "Gold Answer", "Model Wrong Answer"])
t.align = "l"
for i, ex in enumerate(data, 1):
    t.add_row([i, ex["question"], ex["answer"], ex["model_answer"]])
print(t)