#hand_gesture_dectetion.py

if __name__ == '__main__':
    cap = HandGestureDectetion()
    cap.run()
    print('Finish')
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizerOptions, GestureRecognizerResult
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

class HandGestureDectetion:
    def __init__(self):
        self.init_video()
        self.init_mediapipe_detector()

    def init_video(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS,30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    def init_mediapipe_detector(self):
        hand_options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path='./gesture_recognizer.task'),
            running_mode=vision.RunningMode.LIVE_STREAM,
            # num_hands=,
            result_callback=self.on_finish_hands
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.hand_result = None
        # self.finger_index_to_angles = dict()

    def on_finish_hands(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.hand_result = result
        print(f'hand landmark result: consume {self.get_cur_time() - timestamp_ms} ms, {self.hand_result}')
        # if len(self.hand_result.hand_world_landmarks) > 0:
        #     self.calculate_all_fingers_angles()

    # def calculate_all_fingers_angles(self):
    #     hand_landmarks = self.hand_result.hand_world_landmarkers[0]
    #     self.finger_index_to_angles.clear()
    #     for i in range(4):
    #         self.finger_index_to_angles[5 + i * 4] = self.calculate_finger_angle()
    #         self.finger_index_to_angles[6 + i * 4] = self.calculate_finger_angle()
    #         self.finger_index_to_angles[7 + i * 4] = self.calculate_finger_angle()
    #         self.finger_index_to_angles[8 + i * 4] = self.calculate_finger_angle()

    def get_cur_time(self):
        return int(time.time() * 1000)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret == True:
                frame2numpy = np.asarray(frame)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2numpy)
                self.hand_detector.detect_async(mp_image, self.get_cur_time())

                if self.hand_result:
                    frame = self.draw_landmarks_on_image(frame,self.hand_result)

                cv2.imshow('frame',frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        self.exit()

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            ## Get the top left corner of the detected hand's bounding box.
            # height, width, _ = annotated_image.shape
            # x_coordinates = [landmark.x for landmark in hand_landmarks]
            # y_coordinates = [landmark.y for landmark in hand_landmarks]
            # text_x = int(min(x_coordinates) * width)
            # text_y = int(min(y_coordinates) * height) - MARGIN
            #
            # # Draw handedness (left or right hand) on the image.
            # cv2.putText(annotated_image, f"{handedness[0].category_name}",
            #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def exit(self):
        self.cap.release()
        cv2.destroyAllWindows()


#main.py
from hand_gesture_dectetion import HandGestureDectetion

if __name__ == '__main__':
    cap = HandGestureDectetion()
    cap.run()
    print('Finish')

