import numpy as np
import os
import sys
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #logfile = 'INFO2017-12-18T14-45-16.txt20171218-144516.12325'
    logfile = sys.argv[1]
    command_train_loss = 'cat '+logfile + ' | grep \'Train net output #0\' | awk \'{print $11}\''
    command_train_iter = 'cat ' + logfile + ' | grep \'218] Iteration\' | awk \'{print $6}\''
    command_test_loss = 'cat '+logfile + ' | grep \'Test net output #0\' | awk \'{print $11}\''
    command_test_accuracy = 'cat '+logfile + ' | grep \'Test net output #1\' | awk \'{print $11}\''
    command_test_iter = 'cat ' + logfile + ' | grep \'330] Iteration\' | awk \'{print $6}\''
    train_loss = os.popen(command_train_loss).readlines()
    test_loss =  os.popen(command_test_loss).readlines()
    test_accuracy = os.popen(command_test_accuracy).readlines()
    train_iter = os.popen(command_train_iter).readlines()
    test_iter = os.popen(command_test_iter).readlines()
    np_train_loss = [round(float(line.strip()),4) for line in train_loss]
    np_train_iter = [line.strip() for line in train_iter]
    np_test_loss = [round(float(line.strip()),4) for line in test_loss]
    np_test_iter = [line.strip() for line in test_iter]
    np_test_accuracy = [round(float(line.strip()),4) for line in test_accuracy]
    #for i in range(len(np_test_iter)):
        #print(np_test_iter[i])
    #print (len(np_test_loss))
    #print(len(np_test_accuracy))
    plt.figure(1)

    np_test_iter = map(eval, np_test_iter)
    np_train_iter = map(eval, np_train_iter)
    plt.figure(1)
    plt.subplot(221)
    plt.xlabel('iterations')
    plt.ylabel('test_loss')
    plt.title('test loss curse')
    plt.plot(np_test_iter, np_test_loss ,'b',label='test_loss',linewidth = 2) #test loss
    plt.subplot(222)
    plt.xlabel('iterations')
    plt.ylabel('test_accuracy')
    plt.title('test accuracy curve')
    plt.plot(np_test_iter, np_test_accuracy, "b", label="request delay") # test accuracy
    plt.subplot(212)
    plt.plot(np_test_iter, np_test_loss, "b-", label="test_loss")  # test accuracy
    plt.hold
    plt.plot(np_train_iter, np_train_loss, 'r',label="train_loss") #train loss
    plt.xlabel('iterations')
    plt.ylabel('train_loss')
    plt.title('train_loss curve')
    plt.legend()
    plt.show()

