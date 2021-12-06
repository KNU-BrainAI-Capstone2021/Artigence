# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:06:16 2021

@author: PC00
"""

import socket


# 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
HOST = '192.168.0.49' 
# 서버에서 지정해 놓은 포트 번호입니다. 
PORT = 9999       


# 소켓 객체를 생성합니다. 
# 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.  
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# 지정한 HOST와 PORT를 사용하여 서버에 접속합니다. 
client_socket.connect((HOST, PORT))

while True:
    # 메시지를 수신합니다. 
    data = client_socket.recv(1024)
    print('Received', data.decode())

    if len(data.decode())== 0:
        client_socket.close()
# 소켓을 닫습니다.
client_socket.close()