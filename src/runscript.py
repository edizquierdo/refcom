import os
import sys

currentpath = os.getcwd()

for k in range(0,100,2):
    print(k)
    os.system('mkdir '+str(k))
    os.chdir(currentpath+'/'+str(k))
    os.system('time ../main')
    os.chdir(currentpath)
