from pathlib import Path
import json
import time
import random

dmv_path = Path("/mnt/inspurfs/lvkai/llm_data/dmv_wanjuan/")
fps = sorted(list(dmv_path.glob("*/*.jsonl.gz")))

rets = []
for fp in fps:
    with open(fp, "r") as f:
        content = json.load(f)
        rets.append(content)

all_conj = [
    '不料', '确实，', '取决于', '随着', '突然', '而不是', '也许', '这里，', '第三，', '显然', '不管',
    '尽管', '从某种意义上', "某种程度上", '尤其是', '只是', '通常', '假如', '反之，', '有时，', '有时候，', '就像', '结果，',
    "结果就是", '正相反，', '以至于', '目前', '否则，', '除开', '那么，', '也就是说，', '之前', '从另一个角度来看',
    '渐渐地，', '原因是', '可是，', '与此同时，', '不仅如此，', '最后，', '总之，', '换句话说，', '大多数情况下',
    '首先', '不过，', '除此之外', '并且', '同时，', '其次', '当时', '逐渐地', '虽然', '例如，', '起初',
    '而且', '进一步', '不幸地', '简而言之', '可以说', '随后，', '然后，', '总体上，', '很快，', '一方面，',
    '但是，', '总的来说，', '换言之', '这意味着', '特别是', '幸运', '即使', '本质上', '从长远来看', "但事实上，",
    '但', '此后', '当然，', '由此可见，', '以前', '类似地，', '相似地，', '鉴于', '如果', '以及', '因此，', '尽管如此，',
    '事实上，', '由于', '据说', '以便', '无论', '一如既往', '用来', '即便如此', '所以，', '可',
    '之后，', '只要', '此外，', '最初，', '没想到', '此刻', '除非', '反过来，', '另外，', '而是', '或者，',
    '然而，', '最近', '实际上，', '基本上，', '同样地', "同样，", '另一方面，', '接下来', '不论', '既然', '无论如何',
    '这样一来', '乃至于', '相比之下，', '相反', '无疑', '若是', "其中，", "比如，", "综上，",
    "后来，", '而', "不久，", "这时，", "第一，", "第二，", "只有这样，", "于是，", "此处，", "直到目前", "至今为止，",
    "以上就是", "除了", "直到", "最终，", "详细地", "简要地", "惊人地", "历史上", "理论上", "老实说", "一般来说，",
    "通过这样", "理想情况", "[no_conn]"
]

num_conj = {c: 0 for c in all_conj}
print(f"file loaded")
to_write = []
idx_ret = [0 for _ in range(len(rets))]
len_ret = [len(ret) for ret in rets]
total_iter = 0
start_time = time.time()
while True:
    for i, ret in enumerate(rets):
        if idx_ret[i] >= len_ret[i]:
            continue
        item = ret[idx_ret[i]]
        idx_ret[i] += 1
        conn = item["conn"]
        if num_conj[conn] < 12_000:
            to_write.append(item)
            num_conj[conn] += 1
    total_iter += 1
    if total_iter % 10_000 == 0:
        print(f"total_iter: {total_iter}, time: {time.time() - start_time}")
        start_time = time.time()

    if all(num_conj[conn] >= 12_000 for conn in all_conj) or \
            all(idx_ret[i] >= len_ret[i] for i in range(len(rets))):
        break

print("num_conj", num_conj)

exclude_conj = set()
for conn, num in num_conj.items():
    if num < 12_000:
        exclude_conj.add(conn)
        print("exclude_conj", conn, num)

to_write = [item for item in to_write if item["conn"] not in exclude_conj]

random.shuffle(to_write)

train_list = []
dev_list = []
test_list = []

num_conj_train = {c: 0 for c in all_conj if c not in exclude_conj}
num_conj_dev = {c: 0 for c in all_conj if c not in exclude_conj}

for i, item in enumerate(to_write):
    conn = item["conn"]
    if num_conj_train[conn] < 10_000:
        train_list.append(item)
        num_conj_train[conn] += 1
    elif num_conj_dev[conn] < 1_000:
        dev_list.append(item)
        num_conj_dev[conn] += 1
    else:
        test_list.append(item)

write_dict = {
    "train": train_list,
    "dev": dev_list,
    "test": test_list
}

for k in write_dict:
    print(k, len(write_dict[k]))

with open("/mnt/inspurfs/lvkai/llm_data/dmv_wanjuan/merged_data.json", "w+") as f:
    json.dump(write_dict, f, ensure_ascii=False)
