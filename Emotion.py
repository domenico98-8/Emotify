from statistics import mode

import cv2
import dlib
import face_recognition
import numpy as np
from imutils import face_utils
from keras.models import load_model

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.inference import draw_bounding_box
from utils.inference import draw_text
from utils.preprocessor import preprocess_input

USE_WEBCAM = True # If false, loads video file source
def processa():
    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5' #Carica il modello delle emozioni , pre-addestrato
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models #Rilevamento caratteristiche del viso
    detector = dlib.get_frontal_face_detector() #Rilevamento caratteristiche del viso
    emotion_classifier = load_model(emotion_model_path)

    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3] #Fa una stima parziale del modello di input

    # starting lists for calculating modes
    emotion_window = []

    # Initialize some variables
    face_locations = []
    face_encodings = [] #Delimita i volti umani
    face_names = []
    process_this_frame = True


    def face_compare(frame,process_this_frame):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video(cerca i volti nel frame video)
            face_locations = face_recognition.face_locations(rgb_small_frame)#Delimitazione del volto umano
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)#apprende se c'è un volto umano tramite face location

    # starting video streaming
    cv2.namedWindow('window_frame')

    cap = None
    # Webcam source
    cap = cv2.VideoCapture(0)


    while cap.isOpened(): # True:
        ret, frame = cap.read()#restituisce le immagini della webcam frame per frame

        #frame = video_capture.read()[1]

        # To print the facial landmarks
        # landmrk = face_recognition.face_landmarks(frame)
        # for l in landmrk:
        #     for key,val in l.items():
        #         for (x,y) in val:
        #             cv2.circle(frame, (x, y), 1, (255,0, 0), -1)


        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#trasforma i frame in scala di grigio
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#trasforma i frame in rgb

        faces = detector(rgb_image)#rileva caratteristiche del viso
        # face_locations = face_recognition.face_locations(rgb_image)
        # print (reversed(face_locations))
        face_name = face_compare(rgb_image,process_this_frame)
        for face_coordinates ,fname in zip(faces,"Unknow"):
            x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue


            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)#fa delle predizioni
            emotion_probability = np.max(emotion_prediction)#prende la predizione con valore massimo
            emotion_label_arg = np.argmax(emotion_prediction)#prende l'indice della predizione con valore massimo
            emotion_text = emotion_labels[emotion_label_arg]#inserisce nella variabile l'umore predetto
            emotion_window.append(emotion_text)


            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)#restituisce il dato più comune nella lista
            except:
                continue

            if emotion_text == 'Triste':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'Felice':
                color = emotion_probability * np.asarray((255, 255, 0))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()#converte un colore in un elenco del modulo


            name = emotion_text
        
            draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)#crea un rettangolo colorato
            draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, name,color, 0, -45, 0.5, 1)#inserisce l'emozione predetta colorata


        frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#converte da rgb a bgr
        cv2.imshow('window_frame', frame)#mostra il frame nella finestra windows_frame
        if cv2.waitKey(1) & 0xFF == ord('q'):#se c'è un ritardo di un milli secondo allora break
            return emotion_text


    cap.release()
    cv2.destroyAllWindows()

