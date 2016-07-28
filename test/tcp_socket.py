# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:31:27 2016

@author: LY
"""

import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
"""AF_INET指定IPV4协议，AF_INTE6指定IPV6协议，
    SOCK_STREAM指定使用流的TCP协议"""

#建立连接
s.connect(('www.sina.com.cn',80))
#发送数据
s.send('GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')
#接收数据
buffer = []
while True:
    #每次最多接受内容
    d=s.recv(1024)
    if d:
        buffer.append(d)
    else:
        break
data=''.join(buffer)
#关闭连接
s.close()

#将接收到的信息保存
header,html = data.split('\r\n\r\n',1)
print header
#把接收到的信息写入文件
with open('sina.html','wb') as f:
    f.write(html)