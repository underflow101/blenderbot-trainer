import time, os, sys, threading

from flask import Flask, Blueprint, jsonify, request
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

from message_queue import receiveMQ, sendMQ
from configuration import *

model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)

receiveMq = receiveMQ()
sendMq = sendMQ()

api_router = Blueprint('api', __name__, url_prefix='/api/v1')

@api_router.route('/ping', methods=['GET'])
def ping():
    if request.method == 'GET':
        return jsonify({
            'message': "ok",
            'error': None
        })

@api_router.route('/talk', methods=['POST'])
def generate_tensor():
    if request.method == 'POST':
        model_id = request.json['model']
        user_id = request.json['id']
        input_conversation = request.json['conversation']

        if model_id not in MODEL_LIST:
            return jsonify({
                'statusCode': 400,
                'message': "",
                'error': 'Bad Request'
            })
        
        if len(input_conversation) < MIN_LENGTH or len(input_conversation) > MAX_LENGTH:
            return jsonify({
                'statusCode': 400,
                'message': "",
                'error': 'Bad Request'
            })
        
        inputs = tokenizer([input_conversation], return_tensors='pt')
        gen_ids = model.generate(**inputs)
        tmp_ans = tokenizer.batch_decode(gen_ids)
        tmp_ans = ''.join(tmp_ans)
        ans = tmp_ans.replace("<s>", "")
        ans = ans.replace("</s>", "")
        ans = ans.replace("samantha", "Choa")
        ans = ans.replace("Samantha", "Choa")
        ans = ans.strip()

        print("[SERVER] Request complete.")
        return jsonify({
            'statusCode': 200,
            'message': ans,
            'error': None
        })
                

class Api:
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        self.app.config.from_mapping(
            SECRET_KEY="dev"
        )
        self.app.register_blueprint(api_router)
    
    def run(self):
        print("API thread start:", threading.currentThread().getName())
        self.app.run(host=API_URI, port=API_PORT)
        time.sleep(0.05)