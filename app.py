import os
import urllib.request
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import librosa
import numpy as np
import pickle
from werkzeug.utils import secure_filename
from flask import Flask
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] ="C:\\Users\\No One\\Desktop\\ecg_learning"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route('/BT_model', methods=['POST'])
def BT():
    if 'file' not in request.files:
    
    	return{'message' : 'No file part in the request'}
    # resp="hi"
    pre1=[]
    file = request.files['file']
    if file.filename == '':
        return{'message': 'no file selected'}
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  
    f=open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
    sample=pickle.load(f)
    #print(sample)
    pre=np.array(sample)
    pre1.append(pre.T)
    pre1=np.array(pre1)
    model = load_model('0_LSTM_BT_100_final_bt.h5')
    
    for i in range(10):
        start_idx = ((0 * i) + i) * 100
        end_idx = (i + 1) * 100
    
        
        y = model.predict(pre1[:,:,start_idx:end_idx],verbose=0)
        if(max(y[0][0],y[0][1])==y[0][1]):
            return "you have BT"
    return"Norm "        


           

        
       
    #نتيجة المعاالجة
	
	
	
	
	
	
    
    

   

#new route for new model 
@app.route('/MI_model', methods=['POST'])
def MI():
    if 'file' not in request.files:
    
    	return{'message' : 'No file part in the request'}
    # resp="hi"
    pre1=[]
    file = request.files['file']
    if file.filename == '':
        return{'message': 'no file selected'}
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  
    f=open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
    sample=pickle.load(f)
    #print(sample)
    pre=np.array(sample)
    pre1.append(pre.T)
    pre1=np.array(pre1)
    model = load_model('0_LSTM_BT_100_final_mi.h5')
    
    for i in range(10):
        start_idx = ((0 * i) + i) * 100
        end_idx = (i + 1) * 100
    
        
        y = model.predict(pre1[:,:,start_idx:end_idx],verbose=0)
        if(max(y[0][0],y[0][1])==y[0][1]):
            return "you have mi"
    return"Norm "        


           

        
       
    #نتيجة المعاالجة
#st route	
@app.route('/ST_model', methods=['POST'])
def ST():
    if 'file' not in request.files:
    
    	return{'message' : 'No file part in the request'}
    # resp="hi"
    pre1=[]
    file = request.files['file']
    if file.filename == '':
        return{'message': 'no file selected'}
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  
    f=open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
    sample=pickle.load(f)
    #print(sample)
    pre=np.array(sample)
    pre1.append(pre.T)
    pre1=np.array(pre1)
    model = load_model('0_LSTM_window_frequency_100_st_final.h5')
    
    for i in range(10):
        start_idx = ((0 * i) + i) * 100
        end_idx = (i + 1) * 100
    
        
        y = model.predict(pre1[:,:,start_idx:end_idx],verbose=0)
        if(max(y[0][0],y[0][1])==y[0][1]):
            return "you have st"
    return"Norm "        

@app.route('/sound', methods=['POST'])
def sound():
    if 'file' not in request.files:
    
    	return{'message' : 'No file part in the request'}
    # resp="hi"
    
    file = request.files['file']
    if file.filename == '':
        return{'message': 'no file selected'}
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  
    f=open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
    sr=16000
    duration=12
    input_length=16000*12
    best_model_file =load_model( 'best_model_trained_last.hdf5')
    try:
        X, sr = librosa.load( f, sr=sr, duration=duration,res_type='kaiser_fast') 
        dur = librosa.get_duration(y=X, sr=sr)
        if (round(dur) < duration):
              y = librosa.util.fix_length(X, input_length)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
    except Exception as e:
         print("Error encountered while parsing file: ", f)
    feature = (np.array(mfccs).reshape([-1,1]))
    feature = np.expand_dims(feature, axis=0)
    y_pred = best_model_file.predict(feature, batch_size=32)
    classes_x=np.argmax(y_pred,axis=1)
    CLASSES = ['artifact','murmur','normal','extrastole']
    label_to_int = {k:v for v,k in enumerate(CLASSES)}
    int_to_label = {v:k for k,v in label_to_int.items()}
    return int_to_label[classes_x[0]]
    
    return"Norm "        
        




if __name__ == "__main__":
    app.run()    