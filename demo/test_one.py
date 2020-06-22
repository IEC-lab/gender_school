import os
import numpy as np
import cv2


caffe_root = '/home/lc/caffe-drf/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


mean=np.array((127,127,127))


#gender_net_pretrained='./weighted_r18.caffemodel'
gender_net_pretrained='./weighted_caffenet.caffemodel'
#gender_net_model_file='./r18_deploy.prototxt'
gender_net_model_file='./caffenet_deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       input_scale=0.0078125,
                       image_dims=(200, 200))

gender_list=['Male','Female']


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
            img=cv2.resize(img,(200,200))
            image_batch[ix]=img
        batch_predict[begin:end,...]=gender_net.predict(image_batch[:end-begin])
        print ('batch_iter:',end)
    return batch_predict

def predict():
    #prediction_batch=batch_process()
   
    imgpath=('./test2.jpg')
    input_image=cv2.imread(imgpath)
    print(gender_net.predict([input_image])[0].argmax())
   

if __name__=='__main__':
    predict()
