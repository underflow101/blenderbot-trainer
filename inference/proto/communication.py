import websocket
# try:
#     import thread
# except ImportError:
#     import _thread as thread
import os, sys, threading, time
from datetime import datetime
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from message_queue import eagleMQ, commMQ
from configuration import *

class CommSocket(threading.Thread):
    def __init__(self):
        super().__init__()
        websocket.enableTrace(True)
        self.signal_run = False
        self.agent_connected = False
        self.eaglemq = eagleMQ()
        self.commMq = commMQ()
        self.error = None
        
    def push_back_msg(self, msg):
        print("[WS] Input message:", msg)
        print("[WS] Pushing msg to socket...")
        self.ws.send(msg)
    
    def on_message(self, ws, message):
        print("[WS] Message received:", message)
        # self.ws.send("Received Well.")
        self.eaglemq.append_msg(message)
        if message == OP_OFF:
            self.ws.send("ai off")
    
    def on_open(self, ws):
        print("[WS] Socket connected")
        self.agent_connected = True
    
    def on_close(self, ws):
        print("[WS] Socket disconnected")
        self.agent_connected = False
    
    def on_error(self, ws, error):
        print("[WS] [ERROR] Disconnected. Error:", error)
        self.error = error
    
    def run_socket(self):
        def run(*args):
            while True:
                if self.commMq.q:
                    print("[WS] Comm received message(s).")
                    self.push_back_msg(self.commMq.pop_msg())
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