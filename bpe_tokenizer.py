# bpe_tokenizer.py

import re
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(self, num_merges=200):
        self.num_merges = num_merges
        self.vocab = []
        self.token2id = {}
        self.id2token = {}
        self.bpe_ranks = {}
        self.cache = {}
        

    def get_vocab_size(self):
        return len(self.vocab)

    def train(self, text):
        logger.info("Training BPE tokenizer")
        words = text.strip().split()
        vocab = Counter([' '.join(list(word)) + ' </w>' for word in words])
        merges = []

        for _ in range(self.num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges.append(best)
            pattern = re.escape(' '.join(best))
            replace = ''.join(best)
            vocab = Counter({re.sub(pattern, replace, word): freq for word, freq in vocab.items()})

        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        all_tokens = set()
        for word in words:
            tokens = self.bpe(word) 
            all_tokens.update(tokens)

        self.vocab = sorted(all_tokens)
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}

        logger.info("BPE tokenizer training completed")



    def bpe(self, word):
        word_tuple = tuple(word) + ("</w>",)
        if word_tuple in self.cache:
            return self.cache[word_tuple]

        word = word_tuple
        pairs = self.get_pairs(word)

        while True:
            ranked = {pair: self.bpe_ranks[pair] for pair in pairs if pair in self.bpe_ranks}
            if not ranked:
                break

            best = min(ranked, key=ranked.get)
            new_word = []
            i = 0
            while i < len(word):
                j = i
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            pairs = self.get_pairs(word)

        self.cache[word_tuple] = word
        return word


    def get_pairs(self, word):
        """Get symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text):
        encoded = []
        for word in text.strip().split():
            tokens = self.bpe(word)
            for token in tokens:
                if token not in self.token2id:
                    raise KeyError(f"Unknown token after BPE: '{token}'")
                encoded.append(self.token2id[token])
        return encoded


    def decode(self, ids):
        tokens = [self.id2token[i] for i in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def save(self, merge_path):
        with open(merge_path, "w", encoding="utf-8") as f:
            for (a, b), _ in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f"{a} {b}\n")

    def load(self, merge_path):
        with open(merge_path, "r", encoding="utf-8") as f:
            merges = [tuple(line.strip().split()) for line in f if line.strip()]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        logger.info("BPE merges loaded successfully")

    def save_vocab(self, vocab_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(token + "\n")

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f]
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}
        logger.info("BPE vocab loaded successfully")
