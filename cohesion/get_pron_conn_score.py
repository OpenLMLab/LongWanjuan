import json
import glob
import os
import time

import nltk
from nltk import word_tokenize

cn_conj_list = [
    '至今为止，', '目前', '这样一来', '详细地', '与此同时，', '起初', '换言之', '此刻', '鉴于', '其中，', '例如，', '突然', '那么，',
    '不久，', '并且', '确实，', '尽管', '而不是', '总体上，', '第一，', '无论', '最近', '无论如何', '简而言之', '这里，', '有时候，',
    '除非', '结果，', '然后，', '除开', '当然，', '很快，', '但是，', '另一方面，', '换句话说，', '理论上', '历史上', '虽然',
    '不管', '所以，', '首先', '而且', '而', '由于', '第三，', '可是，', '但', '由此可见，', '而是', '最初，', '最终，', '后来，',
    '即使', '只有这样，', '但事实上，', '相反', '总的来说，', '只是', '取决于', '这时，', '用来', '以便', '基本上，',
    '不料', '就像', '接下来', '老实说', '相比之下，', '本质上', '否则，', '从某种意义上', '之前', '当时', '以前', '以至于',
    '特别是', '尤其是', '实际上，', '只要', '理想情况', '或者，', '不仅如此，', '幸运', '事实上，', '然而，', '一方面，', '比如，',
    '通常', '原因是', '从长远来看', '此后', '其次', '渐渐地，', '直到', '不论', '大多数情况下', '之后，', '显然', '也就是说，',
    '以及', '随后，', '没想到', '不过，', '除此之外', '无疑', '第二，', '反过来，', '若是', '以上就是', '也许', '假如',
    '可', '如果', '一如既往', '结果就是', '通过这样', '类似地，', '一般来说，', '除了', '据说', '另外，', '同样地', '反之，',
    '总之，', '进一步', '可以说', '于是，', '最后，', '既然', '尽管如此，', '这意味着', '同时，', '因此，', '某种程度上', '综上，',
    '随着', '此外，', '即便如此', '有时，', '同样，'
]

en_pronoun_list = [
    'one', 'ones', 'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose',
]

cn_pronoun_list = [
    '自己', '我们', '你们', '他们', '她们', '它们', '这些', '这个', '那些', '那个', '那里', '彼此',
    '我', '你', '他', '她', '它', '您', '这', '那',
]


def get_conj_len(text: str):
    conj_len = sum([text.count(conj) * len(conj) for conj in cn_conj_list])
    return conj_len


def get_pronoun_num_en(text: str):
    # text = text.lower()
    word_list = word_tokenize(text)
    pronoun_num = sum([word_list.count(pronoun) for pronoun in en_pronoun_list])
    return pronoun_num, len(word_list)


def get_pronoun_len_cn(text: str):
    pronoun_length = 0
    for pronoun in cn_pronoun_list:
        count = text.count(pronoun)
        if count:
            text = text.replace(pronoun, '*')
            pronoun_length += count * len(pronoun)
    return pronoun_length


data_path = "/cpfs01/shared/public/slimpajama_filter_32k"
info_path = "/cpfs01/shared/public/slimpajama_filter_32k-stat_score"
target_path = "/cpfs01/shared/public/slimpajama_filter_32k-stat_score_plus_cohesion"

fps = sorted(glob.glob(f"{data_path}/*/*.jsonl"))

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"rank: {rank}, world_size: {world_size} {len(fps)}")  # 288
if rank > len(fps):
    exit(0)

fps = fps[rank::world_size]

for fp in fps:
    read_info = json.load(open(fp.replace(data_path, info_path).replace(".jsonl", ".json"), "r"))
    info = {}
    target_path = fp.replace(data_path, target_path).replace(".jsonl", ".json")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    num = 0
    start_time = time.time()
    with open(fp, "r") as f:
        for i in read_info:
            offset = read_info[i]['offset']
            f.seek(offset)
            line = f.readline()
            data_item = json.loads(line)
            id = data_item["id"]
            content = data_item["content"].lower()
            pronoun_num, len_word_list = get_pronoun_num_en(content)

            text_length = len(content)
            info[id] = read_info[i]
            assert info[id]['len_char'] == text_length, f"{info[id]} {text_length}"
            info[id]['pronoun_len'] = pronoun_num
            info[id]['pronoun_rate'] = pronoun_num / len_word_list
            num += 1
            if num % 1000 == 0:
                print(f"{os.path.relpath(fp, data_path)} {num} {time.time() - start_time}s")
                start_time = time.time()

    with open(target_path, 'w') as target_f:
        json.dump(info, target_f, indent=4, ensure_ascii=False)
