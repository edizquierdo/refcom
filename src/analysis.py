import os
import sys

currentpath = os.getcwd()

for k in [79]:
    print(k)
    os.chdir(currentpath+'/'+str(k))
    os.system('time ../simple')
    os.chdir(currentpath)
