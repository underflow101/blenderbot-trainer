import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

if __name__ == '__main__':
    mname = 'facebook/blenderbot-400M-distill'
    # mname = 'hyunwoongko/blenderbot-9B'
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)

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
        ans = ans.strip()
        print("Choa:", ans)
        print("===================================================================================================")