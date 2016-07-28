# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 20:40:31 2016

@author: LY
"""

import socket

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

s.connect(('127.0.0.1',9999))

print s.recv(1024)
#接收欢迎信息
for data in ['liuyuan','junqin','Linda']:
    #接收信息
    s.send(data)
    print s.recv(1024)
s.send('exit')
s.close()
    