#coding=utf-8
import cv2
import datetime
import os
import numpy as np

def relabel(fileName,newfileName):
    inFile=open(fileName,'r')
    outFile=open(newfileName,'w')
    x_lines=inFile.readlines()
    id=0
    before_lable=x_lines[0].split('/')[0]
    for x_line in x_lines:
        x_0=x_line.split()[0]
        current_lable=x_0.split('/')[0]
        if current_lable!=before_lable:
            id=id+1
            outFile.writelines(x_0+' '+np.str(id)+'\n')
            before_lable=current_lable
        else:
            outFile.writelines(x_0+' '+np.str(id)+'\n')
    print(id,'\nDone')

 
def GetFileList(dir_name,fileList):
    newDir = dir_name
    if os.path.isfile(dir_name):
        fileList.append(dir_name)
    elif os.path.isdir(dir_name):  
        for s in os.listdir(dir_name):
            newDir=os.path.join(dir_name,s)
            GetFileList(newDir, fileList)  
    return fileList

def generate_imagelist_from_dir(dir_name):
    filelist=GetFileList(dir_name,[])
    outputfile=open('tmp.txt','w')
    i=0
    for e in filelist:
        #line=e.encode('utf-8').split('/')[-3]+e.encode('utf-8').split('/')[-2]+'/'+e.encode('utf-8').split('/')[-1]+' '+'0'+'\n'
        line=e.split('/')[-2]+'/'+e.split('/')[-1]+' '+'0'+'\n'
        outputfile.writelines(line)
        print ('generating iter:',i)
        i=i+1

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    generate_imagelist_from_dir('/home/lc/gender/school/demo/lc/')
    #generate_imagelist_from_dir('./jitter_v2-align/')
    #relabel('tmp.txt','school_imglist.txt')
    #relabel('all_imagelist_tmp.txt','casia_clean_imagelist.txt')
    #os.remove('all_imagelist_tmp.txt')
    end_time = datetime.datetime.now()
