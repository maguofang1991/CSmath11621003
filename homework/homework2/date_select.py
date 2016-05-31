# -*- coding: utf-8 -*-
#11621003-cshomework2#

file = open("data_list/optdigits.tra")
output = open('data/train.txt', 'w+')
while 1:
    line = file.readline()
    if not line:
        break
    if line[-2] == '3':
        output.write(line[:-3]+"\n")