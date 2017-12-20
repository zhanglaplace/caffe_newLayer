# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:26:38 2017

@author: lk
"""

import sys
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import xlwt
import xlrd

def func_a():
    fd = open('11_data.txt', 'r')
    fs = open('11_save.txt', 'r')
    ftype850_10 = open('./11/ftype850_10.txt', 'w')
    ftype850_25 = open('./11/ftype850_25.txt', 'w')
    ftype850_30 = open('./11/ftype850_30.txt', 'w')
    ftype1300_10 = open('./11/ftype1300_10.txt', 'w')
    ftype1300_25 = open('./11/ftype1300_25.txt', 'w')
    ftype1300_30 = open('./11/ftype1300_30.txt', 'w')
    lines_data = fd.readlines()
    lines_type = fs.readlines()
    cnt = 0

    for line_type in lines_type:
        x = line_type.split('-')[-1].strip()
        if(x == '025M'):
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype850_25.writelines(m+' ')
                ftype850_25.write('\n')
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype1300_25.write(m+' ')
                ftype1300_25.write('\n')

        elif(x == '010M'):
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype850_10.writelines(m+' ')
                ftype850_10.write('\n')
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype1300_10.writelines(m+' ')
                ftype1300_10.write('\n')
        else:
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype850_30.writelines(m+' ')
                ftype850_30.write('\n')
            for i in range(4):
                a = lines_data[cnt]
                b = re.split(',|:|\t', a)[6:-1:2]
                cnt = cnt + 1
                for m in b:
                    ftype1300_30.writelines(m+' ')
                ftype1300_30.write('\n')
    ftype850_25.close()
    ftype850_10.close()
    ftype850_30.close()
    ftype1300_10.close()
    ftype1300_25.close()
    ftype1300_30.close()

def func_excle():
    ftype850_10 = open('./11/ftype850_10.txt', 'r')
    ftype850_25 = open('./11/ftype850_25.txt', 'r')
    ftype850_30 = open('./11/ftype850_30.txt', 'r')
    ftype1300_10 = open('./11/ftype1300_10.txt', 'r')
    ftype1300_25 = open('./11/ftype1300_25.txt', 'r')
    ftype1300_30 = open('./11/ftype1300_30.txt', 'r')
    data_1 = ftype850_10.readlines()
    data_2 = ftype850_25.readlines()
    data_3 = ftype850_30.readlines()
    data_4 = ftype1300_10.readlines()
    data_5 = ftype1300_25.readlines()
    data_6 = ftype1300_30.readlines()
    file = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_1 = file.add_sheet('ftype850_10')
    sheet_2 = file.add_sheet('ftype850_25')
    sheet_3 = file.add_sheet('ftype850_30')
    sheet_4 = file.add_sheet('ftype1300_10')
    sheet_5 = file.add_sheet('ftype1300_25')
    sheet_6 = file.add_sheet('ftype1300_30')
    i = 0
    for line in data_1:
        a= line.split(' ')
        for j in range(4):
            sheet_2.write(i, j, a[j])
        i = i + 1
    i = 0
    for line in data_2:
        a= line.split(' ')
        for j in range(4):
            sheet_3.write(i, j, a[j])
        i = i + 1
    i = 0
    for line in data_3:
        a= line.split(' ')
        for j in range(4):
            sheet_4.write(i, j, a[j])
        i = i + 1
    i = 0
    for line in data_4:
        a= line.split(' ')
        for j in range(4):
            sheet_5.write(i, j, a[j])
        i = i + 1
    i = 0
    for line in data_5:
        a= line.split(' ')
        for j in range(4):
            sheet_6.write(i, j, a[j])
        i = i + 1
    i = 0
    for line in data_6:
        a= line.split(' ')
        for j in range(4):
            sheet_1.write(i, j, a[j])
        i = i + 1
    file.save('./11/result.xls')
    ftype850_25.close()
    ftype850_10.close()
    ftype850_30.close()
    ftype1300_10.close()
    ftype1300_25.close()
    ftype1300_30.close()


def func_b():
    fd = open('11_type.txt', 'r')
    fs = open('11_save.txt', 'w')
    lines_type = fd.readlines()
    fs.write(lines_type[0])
    t = int(lines_type[0].split(':')[0])
    for line_type in lines_type:
        x = int(line_type.split(':')[0])
        if (x - t > 20):
            fs.write(line_type)
            t = x
    fs.close()
    fd.close()


def func_plot(filename):
    fid = open(filename)
    data  = fid.readlines()
    x_1= []
    x_2 = []
    x_3 = []
    x_4 = []
    for line in data:
        a = line.split(' ')
        x_1.append(a[0])
        x_2.append(a[1])
        x_3.append(a[2])
        x_4.append(a[3])
    tmp={}
    for i in x_1:
        if not i in tmp:
            tmp[i] = 1
        else:
            tmp[i] = tmp[i]+1
    p_x = []
    p_y = []
    dict = sorted(tmp.iteritems(), key=lambda asd:asd[0], reverse=False)
    for key in dict:
        #print(key[0])
        p_x.append(key[0])
        p_y.append(key[1])
    plt.plot(p_x,p_y,'--')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('2017-12-10:850-025M')
    plt.show()


if __name__ == '__main__':
    # log_file = '20171210.txt'
    # command_1 = 'cat ' + log_file + ' |grep -n \"n\/a\"'
    # command_2 = 'cat ' + log_file + r' | grep -nE \'010M|025M|030M\''
    # tmp_data = os.popen(command_1).readlines()
    # tmp_type = os.popen(command_2).readlines()
    # print(tmp_data)
    # print(tmp_type)
    # c= 1
    func_plot('./10/ftype850_25.txt')



