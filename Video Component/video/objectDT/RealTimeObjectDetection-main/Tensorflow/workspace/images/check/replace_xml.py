import os
import xml.etree.ElementTree as ET


path  = ('xml')
filenames = os.listdir(path)
for filename in filenames:
    # print(filename)
    r = filename.replace(" ","")
    rx=r.split('.xml')[0]
    rx=rx+'.jpg'
    # print(r)
    txt=filename.split('.')[0]
    # print(txt)

    with open('/home/sachin/Documents/projects/sliit_video/objectDet/real man/RealTimeObjectDetection-main/Tensorflow/workspace/images/check/xml/'+filename, encoding='latin-1') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        tree.find('.//filename').text = rx
        tree.find('.//path').text ='/home/sachin/Documents/projects/sliit_video/objectDet/real man/RealTimeObjectDetection-main/Tensorflow/workspace/images/'+ rx

    tree.write(r, encoding='latin-1')


 