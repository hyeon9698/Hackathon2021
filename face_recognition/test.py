# pip install cmake
# pip install dlib
import face_recognition
import numpy as np
import cv2
from argparse import ArgumentParser
from gaze_tracking import GazeTracking
import keyboard
gaze = GazeTracking()
import cv2
from pynput import keyboard
import time
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator













# current_pressed = set()
# def on_press(key):

#     current_pressed.add(key)
#     print('Key %s pressed' % current_pressed)


# def on_release(key):
#     print('Key %s released' %key)
#     if key == keyboard.Key.esc:
#         return False
#     if key in current_pressed:
#         current_pressed.remove(key)
# print('222hi')
# with keyboard.Listener(
#     on_press=on_press,
#     on_release=on_release) as listener:
#     listener.join()


# print('hi')




# image = face_recognition.load_image_file("your_file.jpg")
# face_landmarks_list = face_recognition.face_landmarks(image)


# picture_of_me = face_recognition.load_image_file("obama.jpg")
# my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# # my_face_encoding은 이제 어느 얼굴과도 비교할 수 있는 내가 가진 얼굴 특징의 보편적인 인코딩을 포함하게 되었습니다. 

# unknown_picture = face_recognition.load_image_file("obama.jpg")
# unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# # 이제 `compare_faces`를 통해 두 얼굴이 같은 얼굴인지 비교할 수 있습니다!

# results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

# if results[0] == True:
#     print("It's a picture of me!")
# else:
#     print("It's not a picture of me!")

cap = cv2.VideoCapture(0)
# # known_image = face_recognition.load_image_file("cap.jpg")
# # dong_encoding = face_recognition.face_encodings(known_image)[0]


# Get the frame size. This will be used by the pose estimator.
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
mark_detector = MarkDetector()




obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]










while True:
    try:
        ret, frame = cap.read()
        if cv2.waitKey(1) == ord('s'):
            print('캡쳐했습니다.')
            cv2.imwrite('dong.jpg', frame)
            dong_image = face_recognition.load_image_file("dong.jpg")
            dong_encoding = face_recognition.face_encodings(dong_image)[0]
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    except:
        print('다시 s 눌러주세요')
        continue

cv2.destroyAllWindows()




















# dong_image = face_recognition.load_image_file("dong.jpg")
# dong_encoding = face_recognition.face_encodings(dong_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    dong_encoding,
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Dong",
    "Barack Obama",
    "Joe Biden"
]
font = cv2.FONT_HERSHEY_DUPLEX
tmp_time = time.time()
while True:
    delay = time.time() - tmp_time
    tmp_time = time.time()
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    frame = rgb_small_frame
    try:
        fps = 1 / delay
    except:
        print('0값임')
        continue
    print("fps: %.2f" % fps)
    # rgb_small_frame = frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    facebox = mark_detector.extract_cnn_facebox(frame)
    if facebox is not None:
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]
            marks = mark_detector.detect_marks(face_img)
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1
            pose = pose_estimator.solve_pose_by_68_points(marks)
            
            if pose[0][0] > 0.5:
                # print('left', pose[0][0])
                cv2.putText(frame, 'left', (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)
                mark_detector.draw_box(frame, [facebox], box_color=(0, 0, 255))
            elif pose[0][0] < -0.5:
                # print('right', pose[0][0])
                cv2.putText(frame, 'right', (x1 + 6, y1 - 6), font, 1.0, (255, 255, 255), 1)
                mark_detector.draw_box(frame, [facebox], box_color=(0, 0, 255))
            if pose[0][1] > 0.3:
                # print('down', pose[0][1])
                cv2.putText(frame, 'down', (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)
                mark_detector.draw_box(frame, [facebox], box_color=(0, 0, 255))
            elif pose[0][1] < -0.3:
                # print('up', pose[0][1])
                cv2.putText(frame, 'up', (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)
                mark_detector.draw_box(frame, [facebox], box_color=(0, 0, 255))

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name == "Dong":               
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            print('부정 행위 LEVEL: 3')
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # results = face_recognition.compare_faces([dong_encoding], unknown_face_encoding)
    # print(results)
    


    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)











    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()