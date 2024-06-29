#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

def main():

    # Argument parsing #################################################################
    dominant_hand='LEFT'

    cap_device = 0
    cap_width = 960
    cap_height = 540

    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)

    keypoint_classifier = KeyPointClassifier()
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    #  ########################################################################
    mode = 0
    pre_processed_landmark_list = None
    tagged_signs = []
    success = False
    after_success = 0
    while True:

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        # hands_results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        left_present = dominant_hand == 'LEFT' and results.left_hand_landmarks is not None
        right_present = dominant_hand == 'RIGHT' and results.right_hand_landmarks is not None
         ####################################################################
        if results.pose_landmarks is not None and left_present or right_present and not success:
            if right_present:
                #right eye edge to thumb tip distance
                x_distance = abs(results.pose_landmarks.landmark[6].x - results.right_hand_landmarks.landmark[4].x)
                y_distance = abs(results.pose_landmarks.landmark[6].y - results.right_hand_landmarks.landmark[4].y)
                brect = calc_bounding_rect(debug_image, results.right_hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    results.right_hand_landmarks.landmark)
            elif left_present:
                #left eye edge to thumb tip distance
                x_distance = abs(results.pose_landmarks.landmark[3].x - results.left_hand_landmarks.landmark[4].x)
                y_distance = abs(results.pose_landmarks.landmark[3].y - results.left_hand_landmarks.landmark[4].y)
                brect = calc_bounding_rect(debug_image, results.left_hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    results.left_hand_landmarks.landmark)
            
            pre_processed_face_landmark_list = pre_process_landmark(
                    results.pose_landmarks.landmark)[:12]
            total_list = pre_processed_landmark_list + pre_processed_face_landmark_list + [x_distance, y_distance]
            # print(total_list)
            hand_sign_id = keypoint_classifier(total_list)
            tagged_signs.append(hand_sign_id)
            # print(tagged_signs)
            if len(tagged_signs) > 30 and tagged_signs.count(0) > 15 and not success:
                after_success += 1
                success = True
                cv.putText(image, "Great Job!", (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                    3.0, (0, 0, 0), 4, cv.LINE_AA)
            elif len(tagged_signs) > 30:
                after_success += 1
                tagged_signs = tagged_signs[-30:]

            # if not success:
            image = draw_info_text(
                image,
                brect,
                dominant_hand,
                keypoint_classifier_labels[hand_sign_id],
                keypoint_classifier_labels,
            )
            image = draw_bounding_rect(use_brect, image, brect, None, keypoint_classifier_labels)

    # Draw pose connections 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                                ) 
        #keep 11 items, 0 to 10 
        # Draw left hand connections
        #gives 21 points
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),  
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2) 
                                    ) 
        # Draw right hand connections 
        #gives 21 points
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,  
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),  
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    ) 
        if success:
            after_success += 1
            if after_success < 30:
                cv.putText(image, "Great Job!", (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                        3.0, (0, 0, 0), 4, cv.LINE_AA)
            if after_success > 40:
                image = cv.putText(image, "Now Closing Window", (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                    3.0, (0, 0, 0), 4, cv.LINE_AA) 
            if after_success > 60:
                break
        cv.imshow('MediaPipe Holistic', image)  # Mirror display)         
        # Screen reflection #############################################################
        
    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def pre_process_landmark(landmark_list):
    x_values = [element.x for element in landmark_list]
    y_values = [element.y for element in landmark_list]

    # temp_landmark_list = copy.deepcopy(landmark_list)
    temp_x = copy.deepcopy(x_values)
    temp_y = copy.deepcopy(y_values)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    index = 0
    for _ in len(temp_x),:
        if index == 0:
            base_x, base_y = temp_x[index], temp_y[index]         
        temp_x[index] = temp_x[index] - base_x
        temp_y[index] = temp_y[index] - base_y
        index += 1
    # Convert to a one-dimensional list
 
    temp_landmark_list = list(itertools.chain(*zip(temp_x, temp_y))) 

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect, hand_sign_id, labels):
    if hand_sign_id == labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     color, 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   labels):
    if hand_sign_text == labels[0]:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 color, -1)

    info_text = handedness
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()
