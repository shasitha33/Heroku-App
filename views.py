from flask import render_template, request
import os
import glob
import cv2
from utils import pipeline_model
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
PREDICT_FOLDER = 'static/predict'
model_gen_final_path = 'model/gender_model.h5'
model_gen = load_model(model_gen_final_path)

def base():
    return render_template('base.html')

def home():
    return render_template('home.html')

def about():
    return render_template('about.html')

def getwidth(path):
    img = cv2.imread(path)
    size = img.shape # height,width & col_layers
    aspect = size[1]/size[0] # width / height
    w = 350 * aspect
    if w>1100: w = 1100
    return int(w)

def gender():
    
    try:
        if request.method == "POST":

            # upfiles = glob.glob(f'{UPLOAD_FOLDER}/*')
            # for f in upfiles:
            #     os.remove(f)

            # predfiles = glob.glob(f'{PREDICT_FOLDER}/*')
            # for f in predfiles:
            #     os.remove(f)

            f = request.files['image']  # check this is image SH
            filename=  f.filename
            path = os.path.join(UPLOAD_FOLDER,filename)
            f.save(path)
            w = getwidth(path)
            # prediction (pass to pipeline model)
            pipeline_model(path,filename,model_gen)
            return render_template('gender.html',fileupload=True,img_name=filename, w=w)
        
        return render_template('gender.html',fileupload=False,img_name="SupunWeerakoon.png")
        
    except:
        return render_template('gender.html',fileupload=False,img_name="SupunWeerakoon.png")