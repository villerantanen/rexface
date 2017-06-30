# all the imports
import sqlite3, time, datetime, os, urllib
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash, redirect, make_response, send_file, \
     send_from_directory
#from revprox import ReverseProxied
import skimage, skimage.transform, skimage.io, skimage.color
import scipy.misc
import pickle
from io import BytesIO
import numpy as np
import face_recognition.api as face_recognition
import sys

# configuration
app = Flask(__name__)
app.config.from_object(__name__)

DUMP_FILE='encodings.pickle'
def read_dump():
    if not os.path.exists(DUMP_FILE):
        FACES={'names':[],'encodings':[],'hashes':[]}
    with open(DUMP_FILE,'rb') as fp:
        FACES=pickle.load(fp)
    return FACES

def write_dump(FACES):
    with open(DUMP_FILE,'wb') as fp:
        pickle.dump(FACES,fp)
    return
FACES=read_dump()
LAST_UPDATE=int(time.time())

@app.before_request
def before_request():
    pass
@app.teardown_request
def teardown_request(exception):
    pass

@app.route('/',methods=['POST','GET','PUT'])
def show_index():
    response_text=""
    if request.method == 'POST':
        submitted_file = request.files['file']
        if submitted_file:
            response_text=get_match()

    return render_template('show_form.html',
                image=response_text)

@app.route('/download_model',methods=['GET'])
def download_model():
    if not os.path.exists(DUMP_FILE):
        return make_response("file not found")
    response=send_from_directory('.',DUMP_FILE)
    return response

@app.route('/read',methods=['GET'])
def read_model():
    FACES=read_dump()
    return make_response("ok")

@app.route('/match',methods=['POST'])
def show_match():
    response_text=get_match()
    response=make_response(response_text)
    response.mimetype="text/plain"
    return response

@app.route('/rex',methods=['POST'])
def show_rex():
    response_text=get_match()
    names="\n".join(
      ['http://rex.reaktor.com/people/'+row.split(",")[0] for row in str(response_text).split("\n")]
    ) 
    response=make_response(names)
    response.mimetype="text/plain"
    return response

@app.route('/train',methods=['POST'])
def train_rex():
    global FACES
    submitted_file = request.files['file']
    submitted_name = request.form['name']
    if not submitted_file:
        return make_response("Image file missing")
    if not submitted_name:
        return make_response("Name for image missing")
    FACES=read_dump()
    image=scipy.misc.imread(submitted_file, mode='RGB')
    size_limit=2500
    if (min(image.shape[0],image.shape[1])>size_limit):
        aspect_ratio = float(image.shape[0])/float(image.shape[1])                   
        if image.shape[0]>image.shape[1]:
            new_height = size_limit
            new_width = int(new_height/aspect_ratio)
        else:
            new_width = size_limit
            new_height = int(aspect_ratio*new_width)
        image = skimage.transform.resize(image, (new_width,new_height))

    new_encoding,message,new_hash=scan_known_people(submitted_name,image,FACES['hashes'])
    if new_hash:
        FACES['names'].append(submitted_name)
        FACES['encodings'].append(new_encoding)
        FACES['hashes'].append(new_hash)
        LAST_UPDATE=int(time.time())
        write_dump(FACES)

    response=make_response(message)
    response.mimetype="text/plain"
    return response


def compute_distances(image_to_check, known_face_encodings):
    unknown_encodings = face_recognition.face_encodings(image_to_check)
    
    return [list(face_recognition.face_distance(known_face_encodings, unknown_encoding)) for unknown_encoding in unknown_encodings]

def find_match(test_image, names, encodings):
    """ return list of names and distances for closest matches """
    matches=[]
    distances=compute_distances(test_image, encodings)
    for distance in distances:
        match_distance=min(distance)
        match_index=distance.index(match_distance)
        matches.append((names[match_index],match_distance))
    return matches

def get_match():
    global LAST_UPDATE
    global FACES
    submitted_file = request.files['file']
    if not submitted_file:
        return "No file submitted"
    # reread dump if old
    if int(time.time())-3600 > LAST_UPDATE:
        print("Reread dump")
        FACES=read_dump()
        LAST_UPDATE=time.time()
    image=scipy.misc.imread(submitted_file, mode='RGB')
    size_limit=2000
    if (min(image.shape[0],image.shape[1])>size_limit):
        aspect_ratio = float(image.shape[0])/float(image.shape[1])                   
        if image.shape[0]>image.shape[1]:
            new_height = size_limit
            new_width = int(new_height/aspect_ratio)
        else:
            new_width = size_limit
            new_height = int(aspect_ratio*new_width)
        image = skimage.transform.resize(image, (new_width,new_height))
    
    matches=find_match(image, FACES['names'], FACES['encodings'])
    if len(matches)==0:
        response_text="noface,1"
    else:
        response_text=[]
        for match in matches:
            response_text.append("%s,%f"%( match[0], match[1]))
        response_text="\n".join(response_text)
    return response_text


def scan_known_people(name,img,existing_hashes=[]):
   
    img_hash=hash(tuple(img.reshape(-1)))
    if img_hash in existing_hashes:
        return False,"Image aready scanned",False
    encodings = face_recognition.face_encodings(img)
    message="Trained"
    if len(encodings) > 1:
        message="WARNING: More than one face found. Only considering the first face."
    if len(encodings) == 0:
        return False,"WARNING: No faces found. Ignoring.",False

    return encodings[0],message,img_hash



def send_image(image):
    byte_io = BytesIO()
    skimage.io.imsave(byte_io, image, plugin='pil', format_str='png')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')



if __name__ == '__main__':   
    app.run()
