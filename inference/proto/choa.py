import threading, time

import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

from message_queue import receiveMQ, sendMQ
from configuration import *

class Choa(threading.Thread):
    def __init__(self):
        super().__init__()
        self.receiveMq = receiveMQ()
        self.sendMq = sendMQ()

        print("[AI] Choa woke up! :)")

    def push_back_msg(self, msg):
        print("[AI] Put message back:", msg)
        self.sendMq.append_msg(msg)

    def load_model(self, mname):
        print("[AI] Loading model...")
        self.model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        print("[AI] Model loaded successfully.")

    def ai_main_loop(self, text):
        print("[AI] tokenizer start")
        inputs = self.tokenizer([text], return_tensors='pt')
        print("[AI] model generate tensor")
        gen_ids = self.model.generate(**inputs)
        print("[AI] batch decode start")
        tmp_ans = self.tokenizer.batch_decode(gen_ids)
        print("[AI] join start")
        tmp_ans = ''.join(tmp_ans)
        print("[AI] replace 1 start")
        ans = tmp_ans.replace("<s>", "")
        print("[AI] replace 2 start")
        ans = ans.replace("</s>", "")
        print("[AI] replace 3 start")
        ans = ans.replace("samantha", "Choa")
        print("[AI] replace 4 start")
        ans = ans.replace("Samantha", "Choa")
        print("[AI] strip start")
        ans = ans.strip()
        print("[AI] Done.")

        print("[AI] Choa's message: " + ans)

        return ans

    def run(self):
        self.load_model(MODEL_NAME)
        print("AI thread start:", threading.currentThread().getName())
        
        while True:
            if self.receiveMq.q:
                msg = self.receiveMq.pop_msg()
                print("[AI] Message:", msg)
                ans = self.ai_main_loop(msg)
                self.push_back_msg(ans)
            time.sleep(0.05)

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