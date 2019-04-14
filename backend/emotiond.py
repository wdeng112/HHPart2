import io
import cv2
import os
from PIL import Image
import os
import speech_recognition as sr
from tqdm import tqdm

from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'client_secret.json'
client = vision.ImageAnnotatorClient()
def detect_faces(path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                   'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('drowsy: {}'.format(likelihood_name[face.anger_likelihood]))
        print('confident: {}'.format(likelihood_name[face.joy_likelihood]))
        print('shocked: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    file = 'live.png'
    cv2.imwrite( file,frame)

    print(detect_faces(file))


    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()