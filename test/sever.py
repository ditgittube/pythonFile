# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 20:04:53 2016

@author: LY
"""

import socket
import threading
import time

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#绑定监听端口
s.bind(('127.0.0.1',9999))
#监听，传入的参数指定等待连接的最大数量
s.listen(5)
print "wait for connection..."


def tcplink(sock,addr):
    print 'Accept new connection from %s:%s...' %addr
    sock.send('Welcome')
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if data == 'exit' or not data:
            break
        sock.send('Hello, %s' % data)
    sock.close()
    print 'Connection from %s:%s closed' % addr

while True:
    #接收一个新连接
    sock,addr = s.accept()
    #创建新线程来处理TCP连接
    t = threading.Thread(target=tcplink,args=(sock, addr))
    t.start()
    

        