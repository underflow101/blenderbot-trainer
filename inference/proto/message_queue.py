'''
message_queue.py

Dev. Dongwon Paek
Executive Dev. Namsu Lee

Message Queue for Internal Communication
'''

from collections import deque

# comm -> AI
class receiveMQ:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            print("[MQ] New receiveMQ created.")
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.q = deque()
    
    def pop_msg(self):
        if self.q:
            print('[MQ] Message:', self.q[0])

            return self.q.popleft()
        else:
            return None
    
    def append_msg(self, msg):
        self.q.append(msg)
        print('[MQ] Saved Message:', msg)
        print('[MQ] Messages in Queue:', self.q)

# AI -> comm
class sendMQ:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            print("[MQ] New sendMQ created.")
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.q = deque()
    
    def pop_msg(self):
        if self.q:
            print('[MQ] Message:', self.q[0])

            return self.q.popleft()
        else:
            return None
    
    def append_msg(self, msg):
        self.q.append(msg)
        print('[MQ] Saved Message:', msg)
        print('[MQ] Messages in Queue:', self.q)
        