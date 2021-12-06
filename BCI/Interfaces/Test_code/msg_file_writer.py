# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:59:07 2021

@author: PC00
"""
while True:
    order = input("1,2,3,exit")
    f = open('C:/Users/PC/Desktop/Server/msg.txt', 'w')
    f.write(order)
    f.close()