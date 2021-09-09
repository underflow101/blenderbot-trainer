import os, sys, threading
from collections import deque

from choa import Choa
from communication import CommSocket

from configuration import *

if __name__ == '__main__':
    agent = Choa()
    comm = CommSocket()
    
    comm.daemon = True
    agent.daemon = True
    
    comm.start()
    agent.start()
    
    comm.join()
    agent.join()