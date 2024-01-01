import base64
import json
import collections
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
app = Flask(__name__)
model=load_model('./model_weights/model_9.h5')
from keras.applications import ResNet50
model2=ResNet50(weights="imagenet",input_shape=(224,224,3))
from keras import Model
new_model=Model(model2.input,model2.layers[-2].output)
def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)#for reshaping
    #Normalization
    img=preprocess_input(img)
    return img
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def predict_caption(photo):
    descriptions = None
    with open('descriptions_1.txt', 'r') as f:
        descriptions = f.read()
    json_string = descriptions.replace("'", "\"")
    descriptions = json.loads(json_string)
    total_words = []
    for key in descriptions.keys():
        [total_words.append(i) for des in descriptions[key] for i in des.split()]
    counter = collections.Counter(total_words)
    freq_count = dict(counter)
    sorted_freq_count = sorted(freq_count.items(), reverse=True, key=lambda x: x[1])
    threshold = 10
    sorted_freq_count = [x for x in sorted_freq_count if x[1] > threshold]
    total_words = [x[0] for x in sorted_freq_count]
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(total_words):
        word_to_idx[word] = i + 1
        idx_to_word[i + 1] = word
    idx_to_word[1846] = "startseq"
    idx_to_word[1847] = "endseq"
    word_to_idx["startseq"] = 1846
    word_to_idx['endseq'] = 1847
    vocab_size = len(word_to_idx) + 1
    maxlen=35

    in_text='startseq'
    for i in range(maxlen):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence],maxlen=maxlen,padding='post')
        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax()#gives word with maximum probability--greedy sampling
        word=idx_to_word[ypred]
        in_text+=' '+word
        if word=='endseq':
            break
    final_caption=in_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    return final_caption
@app.route('/')
def home():
    return render_template('image_captioning.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        app.config['FlickrData'] = 'C:\\Users\\wwwka\\Desktop\\Deep Learning\\Flickr_Data'
        filename=file.filename
        file_path = os.path.join(app.config['FlickrData'], filename)
        file.save(file_path)
        with open(file_path, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
        photo_2048 = encode_image(file_path).reshape((1, 2048))
        pred = predict_caption(photo_2048)
        engine = pyttsx3.init()
        engine.say(pred)
        engine.runAndWait()

        # Pass the audio file path to the HTML template

        return render_template('image_captioning.html', prediction=pred)
    else:
        return render_template('image_captioning.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
