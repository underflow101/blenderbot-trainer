import os, logging

from torch import nn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, PreTrainedTokenizerFast, BlenderbotForConditionalGeneration, BlenderbotModel, BlenderbotConfig, Trainer, TrainingArguments, BlenderbotTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tokenizers import BertWordPieceTokenizer

from config import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ChatDataset(Dataset):
    def __init__(self, filepath, tok_vocab, max_seq_len=128) -> None:
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        self.bos_token = '<s>'      # 0
        self.eos_token = '</s>'     # 2
        self.sep_token = '</s>'     # 2
        self.cls_token = '<s>'      # 0
        self.unk_token = '<unk>'    # 3
        self.pad_token = '<pad>'    # 1
        self.mask_token = '<mask>'  # 4
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_vocab,
            bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
        # self.tokenizer = BlenderbotTokenizer.from_pretrained(tok_vocab, local_files_only=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(tok_vocab, local_files_only=True)
        # tokenizer = BertWordPieceTokenizer(VOCAB_DIR, lowercase=False, strip_accents=False)

    def __len__(self):
        return len(self.data)

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_seq_len:
            while len(input_id) < self.max_seq_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            input_id = input_id[:self.max_seq_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        record = self.data.iloc[index]
        q, a = record['Q'], record['A']
        q_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(a) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            q_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            a_tokens, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_seq_len + 1)])
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                labels += [-100]
        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                'labels': np.array(labels, dtype=np.int_)}

if __name__ == '__main__':
    data_to_show = pd.read_csv(DATA_DIR, encoding='utf-8')
    print(data_to_show.head())
    configuration = BlenderbotConfig(vocab_size=32000,
    max_position_embeddings=128,
    encoder_layers=2,
    encoder_ffn_dim=10240,
    encoder_attention_heads=32,
    decoder_layers=24,
    decoder_ffn_dim=10240,
    decoder_attention_heads=32,
    encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
    use_cache=True,
    is_encoder_decoder=True,
    activation_function='gelu',
    d_model=2560,
    dropout=0.1,
    attention_dropout=0.0,
    activation_dropout=0.0,
    init_std=0.02,
    decoder_start_token_id=1,
    classifier_dropout=0.0,
    scale_embedding=False,
    gradient_checkpointing=False,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    encoder_no_repeat_ngram_size=3,
    forced_eos_token_id=2)
    model = BlenderbotForConditionalGeneration(configuration)
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    model.to(device)
    model.train()

    train_dataset = ChatDataset(DATA_DIR, VOCAB_DIR, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(EPOCHS):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.save_pretrained("./my_blenderbot_model", push_to_hub=False)