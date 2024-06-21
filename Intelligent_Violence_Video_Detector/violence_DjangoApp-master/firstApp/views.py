import glob
import itertools
import json
import os
import random
import shutil
import sys
import time
from time import time
import re
import cv2
from matplotlib.image import thumbnail
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tensorflow import keras
import fileinput
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.eager.context import executing_eagerly_v1
from tensorflow.python.keras.callbacks import TensorBoard
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mp
import speech_recognition as sr
import cv2 
import argparse 
import os 
import pytesseract 
from PIL import Image 
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from pathlib import Path

img_height, img_width=224,224

r = sr.Recognizer()
tokenizer = Tokenizer()
max_length = 20
padding_type='post'
trunc_type='post'
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']

 
#import trained models - using keras load model function
model_video_env=load_model('./models/video/violence_env_classifer.h5')
model_action_env=load_model('./models/video/action_classifer.h5')
model_thumbnail=load_model('./models/thumb/violence_thumb_detector.h5')
model_audio_hate=load_model('./models/audio/hate_detector_audio.h5')
model_subtitle_hate=load_model('./models/subtitle/LSTM_sub.h5')

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("objectdetection_config/pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("objectdetection_config/", 'ckpt-6')).expect_partial()

#label map
category_index = label_map_util.create_category_index_from_labelmap( 'objectdetection_config/label_map.pbtxt')
 

#default calling to home page
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

#reset method
def reset(request):
    path_list=['./media','./data/sound','./data/subtitle','./data/Violence','./audio-chunks']
    context={'a':1}
    print("reset")
    for path in path_list:
          for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    return render(request,'dashboard.html',context)


def about(request):
    return render(request,'about.html')

 
def service(request):
    return render(request,'services.html')


def homepage(request):
    context={'a':1}
    return render(request,'home.html',context)

def viewDataBase(request):
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 


def predictImage(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName='media/'+filePathName  #video save and call each function
    count=extractFrame(filePathName)
    extractSound(filePathName)
    if Path("./data/sound/sound.wav").exists():
        soundToText("./data/sound/sound.wav")
    extractSubtitle()
    context={'filePathName':'filePathName','predictedLabel':'1'}
    return render(request,'dashboard.html',context) 

#main function goes here
def predictVideo(request):
    path  = ('./data/Violence/')
    filenames = os.listdir(path)
    totalImages=len(filenames)
    imageCount=detectEnvironment()          #calling background changes classification method
    actionCount=detectAction()              #calling action classification method
    status=detectThumnail()         
    if(status):
        thumbailStatus="True"
        vilonceThumbnail=1
    else:
        thumbailStatus="False"
        vilonceThumbnail=0
    total_audio,hate_audio=audio_detection()
    total_subtitle,hate_subtitle=subtitle_detection()
    total_object_identified=detectObject()      #calling object detection method

    #final calculation
    if(total_audio>0):
        vilonceAudio=round((hate_audio/total_audio)*100,1)
    else:
        vilonceAudio=0

    if(total_subtitle>0):
        vilonceSubtitle=round((hate_subtitle/total_subtitle)*100)
    else:
        vilonceSubtitle=0
    vilonceVideo=calculateVideoViolence(totalImages,imageCount,actionCount,total_object_identified)         #total images, background changes images, action images, object images
    cumulative,overall=overallViolence(vilonceVideo,vilonceThumbnail,vilonceAudio,vilonceSubtitle)
  
    context={'violenceSummary':overall,'ovrall':cumulative,'vilonceVideo':vilonceVideo,'vilonceThumbnail':vilonceThumbnail,'vilonceSubtitle':vilonceSubtitle,'vilonceAudio':vilonceAudio,'totalImageCount':totalImages,'imageCount':imageCount,'actionCount':actionCount,'objectCount':total_object_identified,'thumbailStatus':thumbailStatus,'totalAudio':total_audio,'hateAudio':hate_audio,'totalsubtitle':total_subtitle,'hateSubtitle':hate_subtitle}
    return render(request,'dashboard.html',context)         #value passing to front end

def calculateVideoViolence(total,env,action,obj):

    mean=total/4
    half=total/2
    if(obj>mean and env>mean and action>mean):     
        ovrall_value=85
    elif(obj>0 and env>half and action>half):   
        ovrall_value=75
    elif(obj>0 and env>mean and action>mean):
        ovrall_value=65
    elif(obj==0 and env>mean and action>mean):
        ovrall_value=25
    else:
        ovrall_value=0
    return ovrall_value

#weight metrix
def overallViolence(video,thumb,audio,subtitle):
    video_weight=0.6
    thumbnail_weight=0.1
    audio_weight=0.15
    subtitle_weight=0.15
    cumulative=round((video_weight*video)+(thumbnail_weight*thumb)+(audio_weight*audio)+(subtitle_weight*subtitle),1)
    
    #over 50% as violence
    if(cumulative>=50):
        ovrall="Violence Video"
    else:
        ovrall="Non Violence Video"
    return cumulative,ovrall


#frame splitting method using OpenCV
def extractFrame(fileName):
    vidcap = cv2.VideoCapture(fileName)
    fname = os.path.basename(fileName).split('.')[0]
    success,image = vidcap.read()
    count = 0
    totalFrames=0
    while success:
        if count % 100 == 0:        #split under 100ms
            totalFrames +=1
            cv2.imwrite("./data/Violence/{}-{}.jpg".format(fname,str(count).zfill(4)),image)     # save frame as JPG file      
        success,image = vidcap.read()
        count += 1
    return totalFrames

#extract sound from video using moviePy library
def extractSound(fileName):
    clip = mp.VideoFileClip(fileName)
    print("info*******")
    print(clip.reader.infos)
    if(clip.reader.infos.get('audio_found')==True):
        clip.audio.write_audiofile("./data/sound/sound.wav")

#extract subtitles from each frame using pytesseract library
def extractSubtitle():
    path  = ('./data/Violence')
    filenames = os.listdir(path)
    for filename in filenames:
        with open('./data/subtitle/extracted_text_subtitle.txt', 'a') as f:
            new_path=path+'/'+filename
            img = Image.open(new_path)
            text = pytesseract.image_to_string(img)
            text_new=cleaner(text)
            if(len(text_new)!=0):
                f.write(text_new)
                f.write('\n')
  

def soundToText(path):
    sound = AudioSegment.from_wav(path)  
    chunks = split_on_silence(sound,
        min_silence_len = 500,
        silence_thresh = sound.dBFS-14,
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("")
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                with open('./data/sound/extracted_text_from_audio.txt', 'a') as f:
                    f.write(text)
                    f.write('\n')
                whole_text += text
    return whole_text


#background changes
def detectEnvironment():
    path  = ('./data/Violence/')
    filenames = os.listdir(path)
    totaDetectedlImage=0
    fireImage=0
    bloodImage=0
    protestImage=0
    for file in filenames:                          #pass each frame to the model 
        img_path='./data/Violence/'+file
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)               #pass image as numpy array
        x = preprocess_input(x)
        result=model_video_env.predict(x)           #keras predict function
        maxElement = np.amax(result)
        # print(maxElement)
        # print(result.argmax())
        if(maxElement>0.9):
            if(result.argmax()==0):
                bloodImage+=1
            elif (result.argmax()==1):
                fireImage+=1
            elif (result.argmax()==2):
                protestImage+=1
    totaDetectedlImage=fireImage+bloodImage+protestImage
    return totaDetectedlImage


#action classification
def detectAction():
    path  = ('./data/Violence/')
    filenames = os.listdir(path)
    totaDetectedlImage=0
    fighting=0
    running=0
    shooting=0
    for file in filenames:                          #pass each frame to the model
        img_path='./data/Violence/'+file
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)               #pass image as numpy array
        x = preprocess_input(x)
        result=model_action_env.predict(x)          #keras predict function
        maxElement = np.amax(result)
        # print(maxElement)
        # print(result.argmax())
        if(maxElement>0.9):
            if(result.argmax()==0):
                fighting+=1
            elif (result.argmax()==1):
                running+=1
            elif (result.argmax()==2):
                shooting+=1
    totaDetectedlImage=fighting+running+shooting
    return totaDetectedlImage

#detection checkpoints
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
#     print(prediction_dict)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

#object detection
def detectObject():
    path  = ('./data/Violence/')
    filenames = os.listdir(path)
    total_objectdetected_images=0
    for file in filenames:
        img_path='./data/Violence/'+file
        img = cv2.imread(img_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

    #creation of bounding box
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.7,
                    agnostic_mode=False,line_thickness=8)
        plt.figure(figsize = (8,90))
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))

        objects = []
        for index, value in enumerate(detections['detection_classes']):
            object_dict = {}
            if detections['detection_scores'][index] > .7:
                object_dict[(category_index.get(value+1)).get('name').encode('utf8')] = detections['detection_scores'][index]
                objects.append(object_dict)
        if(len(objects)>0):
            print(file)
            res = list(objects[0].keys())[0]
            if(str(res).split("b")[1]=="'knife'" or str(res).split("b")[1]=="'pistol'"):            #checking objects
                total_objectdetected_images+=1
                print(str(res).split("b")[1])
    return total_objectdetected_images



def detectThumnail():
    violence=0
    path  = ('./data/Violence/')
    filenames = os.listdir(path)
    file=filenames[0]
    img_path='./data/Violence/'+file
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    result=model_thumbnail.predict(x)
    maxElement = np.amax(result)
    # print(maxElement)
    # print(result.argmax())
    if(maxElement>0.6):
        if(result.argmax()==0):
            violence=0
        elif (result.argmax()==1):
            violence=1
    return violence


def audio_detection():
    hate_speech=0
    total_texts=0
    if Path('./data/sound/extracted_text_from_audio.txt').exists():
        f = open('./data/sound/extracted_text_from_audio.txt')
        for line in f:
            total_texts+=1
            test_text=[]
            test_text.append(line)
            tokenizer.fit_on_texts(test_text)
            post_sequence = tokenizer.texts_to_sequences(test_text)
            padded_post_sequence = pad_sequences(post_sequence, 
                                        maxlen=max_length, padding=padding_type, 
                                        truncating=trunc_type)
            post_prediction = model_audio_hate.predict(padded_post_sequence)
            label = post_prediction.round().item()
            if label == 1:
                hate_speech+=1
   
    return total_texts,hate_speech
    

def subtitle_detection():
    hate_speech=0
    total_texts=0
    if Path('./data/subtitle/extracted_text_subtitle.txt').exists():
        f = open('./data/subtitle/extracted_text_subtitle.txt')
        for line in f:
            total_texts+=1
            test_text=[]
            test_text.append(line)
            tokenizer.fit_on_texts(test_text)
            post_sequence = tokenizer.texts_to_sequences(test_text)
            padded_post_sequence = pad_sequences(post_sequence, 
                                        maxlen=max_length, padding=padding_type, 
                                        truncating=trunc_type)
            post_prediction = model_subtitle_hate.predict(padded_post_sequence)
            label = post_prediction.round().item()
            if label == 1:
                hate_speech+=1
     
    return total_texts,hate_speech
 
def cleaner(word):
  #Remove links
  word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
                '', word, flags=re.MULTILINE)
  #Decontracted words
  word = decontracted(word)
  #Remove users mentions
  word = re.sub(r'(@[^\s]*)', "", word)
  word = re.sub('[\W]', ' ', word)
  #Lemmatized
  list_word_clean = []
  for w1 in word.split(" "):
    if  black_txt(w1.lower()):
      word_lemma =  wn.lemmatize(w1,  pos="v")
      list_word_clean.append(word_lemma)

  #Cleaning, lowering and remove whitespaces
  word = " ".join(list_word_clean)
  if(len(word)==0):
    word="remove"
  word = re.sub('[^a-zA-Z]', ' ', word)
  return word.lower().strip()


def black_txt(token):
  if token == 'u':
    token = 'you'
  return  token not in stop_words_ and token not in list(string.punctuation) and token not in my_sw

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


