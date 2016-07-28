# -*- coding: utf-8 -*-

import socket

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
#SOCK_DGRAM指定Socket的类型为UDP
s.bind(('127.0.0.1',9999))

print 'Bind UDP on 9999'
while True:
    data, addr = s.recvfrom(1024)
    print 'recieved from %s:%s.' % addr
    s.sendto('Hello, %s'% data, addr)
    