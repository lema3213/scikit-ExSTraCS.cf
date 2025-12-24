import re
import pandas as pd

# ====== 1) 读取你的 population CSV ======
csv_path = r"../test/mpr11_export11.csv"
df = pd.read_csv(csv_path)

# ====== 2) 自动识别关键列名（尽量兼容不同导出格式）======
cols = list(df.columns)

def pick_col(prefer_names=(), contains_all=()):
    for n in prefer_names:
        if n in df.columns:
            return n
    for c in cols:
        cl = c.lower()
        if all(k.lower() in cl for k in contains_all):
            return c
    return None

cond_col   = pick_col(prefer_names=("Specified Values", "Code Fragments", "Condition"))
fitness_col= pick_col(prefer_names=("Fitness",), contains_all=("fitness",))
acc_col    = pick_col(prefer_names=("Accuracy",), contains_all=("accuracy",))
match_col  = pick_col(prefer_names=("Match Count",), contains_all=("match","count"))

if cond_col is None:
    cond_col = cols[0]  # 实在找不到就用第一列

# 强制转数值（防止字符串导致 mean 失败）
for c in [fitness_col, acc_col, match_col]:
    if c is not None:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ====== 3) 挑选“较好 classifiers”
# 按你引用的要求：
#   fitness > population 平均 fitness
#   accurate & experienced：这里用 Accuracy > 平均 Accuracy 且 MatchCount > 平均 MatchCount
# ======
avg_fit = df[fitness_col].mean(skipna=True)
avg_acc = df[acc_col].mean(skipna=True)
avg_mc  = df[match_col].mean(skipna=True)

good = df[
    (df[fitness_col] > avg_fit) &
    (df[acc_col] > avg_acc) &
    (df[match_col] > avg_mc)
].copy()

# ====== 4) 提取 CF：去掉 dc、去重、去掉单节点（如 D16 这种）======
bracket_pat = re.compile(r"\[(.*?)\]")
single_node_pat = re.compile(r"^D\d+$", re.IGNORECASE)  # 单节点：D数字

seen = set()
cf_list = []

for cell in good[cond_col].astype(str):
    # 你的 condition 通常是 [cf][dc][cf]... 这种
    frags = bracket_pat.findall(cell)
    if not frags:
        # 如果不是这种格式，就把整格当一个片段（可选）
        frags = [cell]

    for frag in frags:
        frag = frag.strip()
        if not frag:
            continue
        if frag.lower() == "dc":
            continue
        if single_node_pat.fullmatch(frag):  # 去掉单节点 CF
            continue
        if frag not in seen:
            seen.add(frag)
            cf_list.append(frag)

# ====== 5) 输出：无标题头、单列 ======
out_path = r"../MetaData/CF_L2.csv"  # 或 .csv 都行
pd.Series(cf_list).to_csv(out_path, index=False, header=False, encoding="utf-8")

print("selected classifiers:", len(good))
print("final CF count:", len(cf_list))
print("saved to:", out_path)
