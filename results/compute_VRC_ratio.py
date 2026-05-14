import json
from collections import Counter

file_path = "/root/I-DLM/results/draft_conf_N4.jsonl"

counter = Counter()
total = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            case = data.get("case")
            if case in ["V", "R", "C"]:
                counter[case] += 1
                total += 1
                if case == "R":
                    counter["R_accept_tokens"] += data.get("accept_len")
        except json.JSONDecodeError:
            continue  # 忽略非 JSON 行

# 输出比例
for k in ["V", "R", "C"]:
    print(f"{k}: {counter[k]} ({counter[k]/total:.2%})")
    
print(f'{counter["R_accept_tokens"]/ counter["R"]:.3f}')