import os, sys, threading, argparse
from collections import deque

from choa import Choa
from communication import CommSocket
from api import Api

from configuration import *

def socket_mode():
    agent = Choa()
    comm = CommSocket()

    comm.daemon = True
    agent.daemon = True

    agent.start()
    comm.start()

    agent.join()
    comm.join()

def api_mode():
    api = Api()
    api.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ws' ,'--websocket', dest='mode_ws', action='store_true', help='Websocket Mode')
    parser.add_argument('-api', '--api_server', dest='mode_api', action='store_true', help='API Server Mode')
    
    args = parser.parse_args()

    if args.mode_ws:
        socket_mode()
    elif args.mode_api:
        api_mode()