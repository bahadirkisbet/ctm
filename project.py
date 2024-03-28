import cv2

import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dilbilim.project.utils import get_all_files

HandJoins = mp.solutions.hands.HandLandmark


class Pipeline:

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.files = get_all_files(base_path)[:10]
        self.ref_landmarks = [
            HandJoins.INDEX_FINGER_TIP, HandJoins.MIDDLE_FINGER_TIP,
            HandJoins.RING_FINGER_TIP, HandJoins.PINKY_TIP,
            HandJoins.THUMB_TIP, HandJoins.WRIST
        ]

    def get_all_pose_coordinates(self, files):
        all_results = []
        for file in files:
            all_results.append(self.get_pose_coordinates(file))
        return all_results

    def get_pose_coordinates(self, video_path):
        camera = cv2.VideoCapture(video_path)

        # Read first frame for initialization
        _, frame = camera.read()

        # Mediapipe (POSE ESTIMATION)
        mp_holistic = mp.solutions.holistic

        results = []
        with mp_holistic.Holistic(min_detection_confidence=0.01, min_tracking_confidence=0.01) as holistic:
            while True:
                try:
                    # Extract joint information
                    temp = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results.append(temp)
                    ret, frame = camera.read()

                except Exception as _:
                    break

        # Release resources
        camera.release()
        cv2.destroyAllWindows()

        return results

    def get_positions(self, all_landmarks):
        # take average of all tips
        temp_list_x = []
        temp_list_y = []
        temp_list_z = []
        for ref in self.ref_landmarks:
            temp_list_x.append(all_landmarks[ref].x)
            temp_list_y.append(all_landmarks[ref].y)
            temp_list_z.append(all_landmarks[ref].z)
        temp_position_x = sum(temp_list_x) / len(temp_list_x)
        temp_position_y = sum(temp_list_y) / len(temp_list_y)
        temp_position_z = sum(temp_list_z) / len(temp_list_z)
        return temp_position_x, temp_position_y, temp_position_z

    def process_landmark(self, cv2_result: list):
        temp_reference_x = []
        temp_reference_y = []
        temp_reference_z = []

        for frame in cv2_result:
            if frame.right_hand_landmarks:
                pos_x, pos_y, pos_z = self.get_positions(frame.right_hand_landmarks.landmark)
                temp_reference_x.append(pos_x)
                temp_reference_y.append(pos_y)
                temp_reference_z.append(pos_z)

            else:
                temp_reference_x.append(0)
                temp_reference_y.append(0)
                temp_reference_z.append(0)

        return temp_reference_x, temp_reference_y

    def extract_landmarks(self, cv2_result: list):
        temp_reference_x = []
        temp_reference_y = []
        temp_reference_z = []

        for frame in cv2_result:  # FIXME: check if the landmarks are provided from the same organ.
            if frame.right_hand_landmarks:
                pos_x, pos_y, pos_z = self.get_positions(frame.right_hand_landmarks.landmark)
                temp_reference_x.append(pos_x)
                temp_reference_y.append(pos_y)
                temp_reference_z.append(pos_z)

            else:
                temp_reference_x.append(0)
                temp_reference_y.append(0)
                temp_reference_z.append(0)

        return temp_reference_x, temp_reference_y, temp_reference_z

    def process_videos(self):
        all_results = self.get_all_pose_coordinates(self.files)

        for result, file_name in zip(all_results, self.files):
            landmark = self.extract_landmarks(result)

            # 1D list
            landmark_x = landmark[0]
            landmark_y = landmark[1]
            landmark_z = landmark[2]

            # 2D list
            landmark_xy = [landmark_x, landmark_y]
            landmark_xz = [landmark_x, landmark_z]
            landmark_yz = [landmark_y, landmark_z]

            # 3D list
            landmark_xyz = [landmark_x, landmark_y, landmark_z]

            # Calculate velocity
            velocity_x = self.calculate_velocity([landmark_x])
            velocity_y = self.calculate_velocity([landmark_y])
            velocity_z = self.calculate_velocity([landmark_z])
            velocity_xy = self.calculate_velocity(landmark_xy)
            velocity_xz = self.calculate_velocity(landmark_xz)
            velocity_yz = self.calculate_velocity(landmark_yz)
            velocity_xyz = self.calculate_velocity(landmark_xyz)

            df = pd.DataFrame()

            # Append to dataframe
            df["velocity_x"] = velocity_x
            df["velocity_y"] = velocity_y
            df["velocity_z"] = velocity_z
            df["velocity_xy"] = velocity_xy
            df["velocity_xz"] = velocity_xz
            df["velocity_yz"] = velocity_yz
            df["velocity_xyz"] = velocity_xyz

            f, axs = plt.subplots(figsize=(15, 10))
            for col in df.columns:
                axs.plot(range(1, len(velocity_x) + 1), df[col], label=col)
                axs.legend()
            temp = file_name.split('/')
            verb_type = temp[-2]
            raw_file_name = temp[-1].split('.')[0]
            file_path = f"output/{verb_type}_{raw_file_name}.png"
            plt.savefig(file_path)

    def calculate_velocity(self, arrays):
        velocity = []
        length = len(arrays[0])
        ref_point_count = len(arrays)
        for i in range(1, length):
            prev_position = [arrays[j][i - 1] for j in range(ref_point_count)]
            curr_position = [arrays[j][i] for j in range(ref_point_count)]
            velocity.append(np.sqrt(sum([(curr_position[j] - prev_position[j]) ** 2 for j in range(ref_point_count)])))
        return velocity


# file_names = ['video_1.mp4', 'video_2.mp4']
# folder_path = 'videos'

def main():
    pipeline = Pipeline('Videos')
    pipeline.process_videos()


if __name__ == '__main__':
    main()
