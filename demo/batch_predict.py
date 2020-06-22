import os
import numpy as np
import cv2
import time

caffe_root = '/home/lc/caffe-drf/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


#mean_filename='./mean.binaryproto'
#proto_data = open(mean_filename, "rb").read()
#a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
#mean  = caffe.io.blobproto_to_array(a)[0]
#m_min, m_max = mean.min(), mean.max()
#normal_mean = (mean - m_min) / (m_max - m_min)
#in_shape=(200, 200)
#mean = caffe.io.resize_image(normal_mean.transpose((1,2,0)),in_shape).transpose((2,0,1)) * (m_max - m_min) + m_min
mean=np.array((127,127,127))


#gender_net_pretrained='./weighted_r18.caffemodel'
#gender_net_model_file='./r18_deploy.prototxt'
gender_net_pretrained='./weighted_caffenet.caffemodel'
gender_net_model_file='./caffenet_deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       input_scale=0.0078125,
                       image_dims=(200, 200))

gender_list=['Male','Female']

IMAGE_LIST_FILE='/home/lc/data/gender_200w/align_school/val.txt'
DATA_ROOT='/home/lc/data/gender_200w/align_school/'
BATCH_SIZE=200
total_time=0

def read_image_list():
    print('read image list: %s' % IMAGE_LIST_FILE)
    image_list = []
    with open(IMAGE_LIST_FILE, 'r') as f:
        image_list = [line.split()[0] for line in f.readlines()]
    return image_list


def batch_process():
    image_list=read_image_list()
    image_shape=(200,200,3)
    data_shape=(BATCH_SIZE,)+image_shape
    image_batch=np.zeros(np.array(data_shape),dtype=np.float32)
    num=len(image_list)
    batch_predict=np.zeros(np.array((num,2)),dtype=np.float32)
    for begin in range(0,num,BATCH_SIZE):
        end =min(begin+BATCH_SIZE,num)
        batch_list=image_list[begin:end]
        for ix,image in enumerate(batch_list):
            img=cv2.imread(DATA_ROOT+image.split()[0])
            image_batch[ix]=img
        tic=time.time()
        batch_predict[begin:end,...]=gender_net.predict(image_batch[:end-begin])
        toc=time.time()
        total_time=toc-tic
        print ('batch_iter:',end)
    return batch_predict

def predict():
    prediction_batch=batch_process()
    lines=open(IMAGE_LIST_FILE,'r')
    i=0
    correct=0
    F_Male=0
    F_Female=0
    male_num=0
    female_num=0
    for line in lines:
        imgpath=DATA_ROOT+line.split()[0]
        label_gender=int(line.split()[1])
        prediction_gender=prediction_batch[i].argmax()
        if label_gender==0:
            male_num+=1
        else:
            female_num+=1
        if prediction_gender==label_gender:
            correct+=1
        if prediction_gender==0 and label_gender==1:
            F_Female+=1
            os.system("cp "+imgpath+" /home/lc/gender/school/demo/mis_female/")
        elif prediction_gender==1 and label_gender==0:
            F_Male+=1
            os.system("cp "+imgpath+" /home/lc/gender/school/demo/mis_male/")
        i=i+1
        print ('iter:',i)
    print("Female error:{} ({}/{})".format(float(F_Female/female_num),F_Female,female_num))
    print("Male error:{} ({}/{})".format(float(F_Male/male_num),F_Male,male_num))
    print("Accuracy:",float(correct)/i)
    print("Correct:",correct)
    print("Total:",i)
    print("Speed:",float(total_time)/i)
    print ('Done!')

if __name__=='__main__':
    predict()
