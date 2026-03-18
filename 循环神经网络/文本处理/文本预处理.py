import collections
import re

def tokenize(lines, token='word'):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("Error:未知令牌类型：" + token)

def count_corpus(tokens):
    """统计标记的频率：这里的tokens是1D列表或者2D列表"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将tokens展平成使用标记填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """构建文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 根据出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知标记的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()  # 根据索引找标记和根据标记找索引
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """转换到一个一个的item进行输出"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """如果是单个index直接输出，如果是list或者tuple迭代输出"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0],list):
        tokens = [tokens for line in tokens for tokens in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回Time machine数据集中的标记索引列表和词汇表"""
    lines = read_time_machine()
    tokens = tokenize(lines,'char')
    vocab = Vocab(tokens)
    # 因为Time machine数据集中每一个文本行，不一定是一个句子或者段落
    # 所以将所有文本行展平到一个列表之中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens >0:
        corpus = corpus[:max_tokens]
    return corpus,vocab

corpus,vocab = load_corpus_time_machine()
len(corpus),len(vocab)
