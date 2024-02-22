import gzip
import json
import time
from pathlib import Path
import os
from random import random

from ltp import StnSplit

def start_with_conj(sentence, conj):
    for c in conj:
        if sentence.startswith(c):
            return c
    return None

ltp_sp = StnSplit()

wanjuan_path = Path("/mnt/inspurfs/share_data/llm_data/wanjuan/nlp/CN")
dmv_path = Path("/mnt/inspurfs/lvkai/llm_data/dmv_wanjuan/")
rank, world_size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
fps = sorted(list(wanjuan_path.glob("*/*.jsonl.gz")))
if rank == 0:
    print(f"file num: {len(fps)}")
if rank >= len(fps):
    exit(0)
if world_size < len(fps):
    raise ValueError(f"rank: {rank}, world_size: {world_size}, len(fps): {len(fps)}")
fp = fps[rank]

os.makedirs(dmv_path.joinpath(fp.relative_to(wanjuan_path)).parent, exist_ok=True)
dmv_file_path = dmv_path.joinpath(fp.relative_to(wanjuan_path))
data_file = gzip.open(fp, 'r')

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
    "通过这样", "理想情况"
]

num_conj = {c: 0 for c in all_conj}
nun_no_conj = 0
start_time = time.time()
num = 0
ret = []
while True:
    try:
        data_line = data_file.readline()
        data_dict = json.loads(data_line)
        content = data_dict['content']

    except Exception as e:
        print(e)
        print(f"{fp.relative_to(wanjuan_path)}: {num_conj} done")
        with open(dmv_file_path, 'w+') as fp:
            json.dump(ret, fp, ensure_ascii=False)
        break

    sentences = ltp_sp.split(content)
    sentences = [i for i in sentences if 5 <= len(i) <= 128]
    for i, sentence in enumerate(sentences):
        conj = start_with_conj(sentence, all_conj)
        if conj and i >= 1:
            ret.append({
                "conn": conj,
                "arg1": sentences[i - 1],
                "arg2": sentence[len(conj):],
            })
            num_conj[conj] += 1
            if num_conj[conj] == 500:
                all_conj = [x for x in all_conj if x != conj]
        elif nun_no_conj < 500 and i >= 1:
            rand_v = random()
            if rand_v < 0.01:
                ret.append({
                    "conn": "[no_conn]",
                    "arg1": sentences[i - 1],
                    "arg2": sentence,
                })
                nun_no_conj += 1

    num += 1

    if num % 500000 == 0 or len(all_conj) == 0:
        print(f"{fp.relative_to(wanjuan_path)}: {num} {time.time() - start_time}s")
        start_time = time.time()

    if len(all_conj) == 0:
        print(f"{fp.relative_to(wanjuan_path)}: {num_conj} done")
        with open(dmv_file_path, 'w+') as fp:
            json.dump(ret, fp, ensure_ascii=False)
        break
