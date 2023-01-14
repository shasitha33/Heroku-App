import numpy as np
import cv2

import warnings
warnings.filterwarnings('ignore')

def final_face_cordinates(gray_image):
      
  haar_front = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
  haar_profile = cv2.CascadeClassifier('model/haarcascade_profileface.xml')
  
  af1_list = [1.5,1.2, 1.3]
  af2_list = [5,2,  3]
  ap1_list = [1.5,1.4, 1.2]
  ap2_list = [5,3,  3]
  face_cordinates = []

  for i in range(len(af1_list)):

    array_front = haar_front.detectMultiScale(gray_image,af1_list[i],af2_list[i])
    array_profile = haar_profile.detectMultiScale(gray_image,ap1_list[i],ap2_list[i])
    
    if (len(array_front) == 0) and (len(array_profile) > 0): temp_face_cordinates = [list(x) for x in array_profile]
    elif len(array_profile) == 0: temp_face_cordinates = [list(x) for x in array_front]
    else: temp_face_cordinates = [list(x) for x in np.concatenate((array_front, array_profile), axis=0)]
    face_cordinates.extend(temp_face_cordinates)

  d_face_cords = {(x,y,w,h):True for x,y,w,h in face_cordinates}
  face_cordinates= [[x,y,w,h] for x,y,w,h in d_face_cords.keys()]
  d_keep = {id:True for id in range(len(face_cordinates))}
  d_xy = {id:(item[0],item[0]+item[2],item[1],item[1]+item[3]) for id,item in enumerate(face_cordinates)}

  for id_keep,item_keep in enumerate(face_cordinates):
    x_lu = item_keep[0] ; y_lu = item_keep[1];
    x_ld = item_keep[0] ; y_ld = item_keep[1] + item_keep[3];
    x_ru = item_keep[0] + item_keep[2] ; y_ru = item_keep[1];
    x_rd = item_keep[0] + item_keep[2] ; y_rd = item_keep[1] + item_keep[3];
    for i in range(len(face_cordinates)):
      x_min = d_xy[i][0]; x_max = d_xy[i][1];
      y_min = d_xy[i][2]; y_max = d_xy[i][3];

      p = 0.3
      px_max = x_max - (x_max-x_min)*p ; px_min = x_min + (x_max-x_min)*p ;
      py_max = y_max - (y_max-y_min)*p ; py_min = y_min + (y_max-y_min)*p ;

      if id_keep==i or not d_keep[i]: continue
      elif (x_min <= x_lu <= px_max) and (y_min <= y_lu <= py_max):
            d_keep[id_keep] = False
            continue
      elif (x_min <= x_ld <= px_max) and (py_min <= y_ld <= y_max):
            d_keep[id_keep] = False
            continue 
      elif (px_min <= x_ru <= x_max) and (y_min <= y_ru <= py_max):
            d_keep[id_keep] = False
            continue 
      elif (px_min <= x_rd <= x_max) and (py_min <= y_rd <= y_max):
            d_keep[id_keep] = False
            continue 

  return [list(face_cordinates[id]) for id in range(len(face_cordinates)) if d_keep[id]]

def Crop_Load(image_path, save_path, model_gen):
  
  image_upload = cv2.imread(image_path)
  gray_image = cv2.cvtColor(image_upload,cv2.COLOR_BGR2GRAY)
  crop_images = []
  face_cordinates = final_face_cordinates(gray_image)

  for x,y,w,h in face_cordinates:
    roi = gray_image[y:y+h,x:x+w]
    crop_images.append(roi)
  
  if len(crop_images)!=0:
    for id,im in enumerate(crop_images):  
      if im.shape[0] >= 48:  # Minimum size of acceptable image is 48
        im_48 = cv2.resize(im,(48,48),cv2.INTER_AREA) # Shrink image to 48 x 48
        im_ch3 = cv2.cvtColor(im_48,cv2.COLOR_GRAY2RGB) # Gray image with 3 channels
        im_reshape = im_ch3.reshape(1,48,48,3) # batch_size,height,width,channels
        im_scaled = im_reshape/255  # We trained model for scaled data
        model_pred = round(float(model_gen.predict(im_scaled)[0][0]),3) #python cannot round np.float32

        if model_pred <= 0.30: gender = 'Female';
        elif model_pred >=0.70: gender = 'Male'; 
        else: continue

        x,y,w,h = face_cordinates[id]
        cv2.rectangle(image_upload,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(image_upload, f'{gender}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      
    cv2.imwrite(save_path, image_upload)

def pipeline_model(path,filename,model_gen):
      save_path = f'static/predict/{filename}'
      image_path = path
      Crop_Load(image_path, save_path, model_gen)
      


