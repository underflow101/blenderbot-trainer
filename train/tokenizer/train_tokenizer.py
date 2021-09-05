from pathlib import Path
import os

from tokenizers import BertWordPieceTokenizer
from Korpora import Korpora

from config import *

if __name__ == '__main__':
    tokenizer = BertWordPieceTokenizer(
        #vocab_file=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
    )

    tokenizer.train(
        files=[DATA_DIR],
        limit_alphabet=6000,
        vocab_size=32000,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
    )

    tokenizer.save('./vocab.txt')
    output = tokenizer.encode("<s>안녕 찌찌야!</s>")
    print(output.ids)
    print(output.tokens)
    print(tokenizer.decode(output.ids))
