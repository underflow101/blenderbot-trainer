import os, sys, threading, time

import websocket

from message_queue import receiveMQ, sendMQ
from configuration import *

class CommSocket(threading.Thread):
    def __init__(self):
        super().__init__()
        websocket.enableTrace(False)
        self.signal_run = False
        self.agent_connected = False
        self.receiveMq = receiveMQ()
        self.sendMq = sendMQ()
        self.error = None
        
    def push_msg_to_socket(self, ws, msg):
        print("[WS] Input message:", msg)
        print("[WS] Pushing msg to socket...")
        self.ws.send(msg)
    
    def on_message(self, ws, message):
        print("[WS] Message received:", message)
        self.receiveMq.append_msg(message)
    
    def on_open(self, ws):
        print("[WS] Socket connected")
        self.ws.send("hey :D 1")
        self.ws.send("hey :D 2")
        self.ws.send("hey :D 3")
        
        self.agent_connected = True
    
    def on_close(self, ws, close_status_code, close_msg):
        print("[WS] Socket disconnected")
        self.agent_connected = False
    
    def on_error(self, ws, error):
        print("[WS] [ERROR] Disconnected. Error:", error)
        self.error = error
    
    def run_socket(self):
        def run(*args):
            while True:
                if self.sendMq.q:
                    print("[WS] Comm received message(s).")
                    self.push_msg_to_socket(self.sendMq.pop_msg())
                if self.error != None:
                    print("[WS] Error occurred:", self.error)
                    self.error = None
                time.sleep(0.05)
        threading.Thread(target=run).start()
        time.sleep(0.05)
    
    def run(self):
        print("Comm thread start:", threading.currentThread().getName())
        self.ws = websocket.WebSocketApp(WS_URI,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.run_socket()
        while True:
            self.ws.run_forever()
            time.sleep(0.05)