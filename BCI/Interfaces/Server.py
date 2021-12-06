# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:55:01 2021

@author: PC00
"""

import socket

HOST = '192.168.0.49'
PORT = 9999

print("a")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("a")
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
print("a")
server_socket.bind((HOST,PORT))
print("a")
server_socket.listen()
print("a")
client_socket, addr = server_socket.accept()
print("Connected by", addr)

copy = '0'
print("a")
while True:
    order = input("1,2,3,exit")
    if(order == 'end') ==1:
        client_socket.sendall(order.encode())
        break
    f = open('C:/Users/PC/Desktop/Server/msg.txt', 'w')
    f.write(order)
    f.close()
    
    f = open('C:/Users/PC/Desktop/Server/msg.txt', 'r')
    s = f.read()
    print("send msg", s)
    copy = s
    client_socket.sendall(s.encode())


client_socket.close()
server_socket.close()
