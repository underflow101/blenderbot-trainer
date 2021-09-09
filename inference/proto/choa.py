import threading

import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

from message_queue import receiveMQ, sendMQ
from configuration import *

class Choa(threading.Thread):
    def __init__(self):
        super().__init__()
        self.receiveMq = receiveMQ()
        self.sendMq = sendMQ()

    def push_back_msg(self, msg):
        self.sendMq.append_msg(msg)

    def load_model(self, mname):
        self.model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    def ai_main_loop(self, text):
        inputs = self.tokenizer([text], return_tensors='pt')
        gen_ids = self.model.generate(**inputs)
        tmp_ans = self.tokenizer.batch_decode(gen_ids)
        tmp_ans = ''.join(tmp_ans)
        ans = tmp_ans.replace("<s>", "")
        ans = ans.replace("</s>", "")
        ans = ans.replace("samantha", "Choa")
        ans = ans.replace("Samantha", "Choa")
        ans = ans.strip()

        return ans

    def run(self):
        self.load_model(MODEL_NAME)
        
        while True:
            if self.receiveMq.q:
                ans = self.ai_main_loop(self.receiveMq.pop_msg())
                self.push_back_msg(ans)

if __name__ == '__main__':
    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)

    while True:
        print("===================================================================================================")
        text = input("You: ").strip()
        print("===================================================================================================")
        inputs = tokenizer([text], return_tensors='pt')
        gen_ids = model.generate(**inputs)
        tmp_ans = tokenizer.batch_decode(gen_ids)
        tmp_ans = ''.join(tmp_ans)
        ans = tmp_ans.replace("<s>", "")
        ans = ans.replace("</s>", "")
        ans = ans.replace("samantha", "Choa")
        ans = ans.replace("Samantha", "Choa")
        ans = ans.strip()
        print("Choa:", ans)
        print("===================================================================================================")