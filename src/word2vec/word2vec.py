from gensim.models import KeyedVectors

# 加载词向量模型
model = KeyedVectors.load_word2vec_format(
    "../Dataset/wiki_word2vec_50.bin", binary=True
)

word2index = {"<PAD>": 0, "<UNK>": 1}
with open("../Dataset/train.txt", "r", encoding="utf-8") as f:
    for line in f:
        words = line.split()[1:]
        for word in words:
            if not word in word2index.keys():
                word2index[word] = len(word2index)
