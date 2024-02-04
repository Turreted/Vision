import argparse
from time import sleep
from time import time
import os

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import threading

import face_recognition


class StreamViewer:
    def __init__(self, port=PORT):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind("tcp://*:" + port)
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode_(""))
        self.current_frame = None
        self.keep_running = True
        print(f"bound to port {port}")

        context = zmq.Context()
        self.resp_socket = context.socket(zmq.PUB)
        self.resp_socket.connect(
            "tcp://" + "raspberrypi.local" + ":" + str(int(port) + 1)
        )  # TODO: might cause bugs
        os.chdir("known_faces")

        self.known_face_encodings = []
        self.known_face_names = []

        for img_name in os.listdir("."):
            self.known_face_encodings.append(
                face_recognition.face_encodings(
                    face_recognition.load_image_file(img_name)
                )[0]
            )
            self.known_face_names.append(img_name[:-4])

        os.chdir("../")
        print(f"known face names ", self.known_face_names)

        # start openCV thread
        x = threading.Thread(target=self.send_info, args=())
        x.start()

    def receive_stream(self, display=True):
        """
        Displays displayed stream in a window if no arguments are passed.
        Keeps updating the 'current_frame' attribute with the most recent frame, this can be accessed using 'self.current_frame'
        :param display: boolean, If False no stream output will be displayed.
        :return: None
        """
        self.keep_running = True
        while self.footage_socket and self.keep_running:
            try:
                frame = self.footage_socket.recv_string()
                self.current_frame = string_to_image(frame)

                if display:
                    cv2.imshow("Stream", self.current_frame)
                    cv2.waitKey(1)

            except KeyboardInterrupt:
                self.keep_running = False
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def send_info(self):
        print("this is a test")
        sleep(1)
        i = 0

        while self.resp_socket and self.keep_running:
            if len(self.current_frame) == 0: continue

            small_frame = cv2.resize(self.current_frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame, 2)

            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding
                )
                name = "Unknown"

                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)

            print(face_names)                

            try:
                self.resp_socket.send_string(f"dummy data {i}")
            except KeyboardInterrupt:
                self.keep_running = False
                break

            i += 1

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False


def main():
    port = PORT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        help="The port which you want the Streaming Viewer to use, default"
        " is " + PORT,
        required=False,
    )

    args = parser.parse_args()
    if args.port:
        port = args.port

    stream_viewer = StreamViewer(port)
    stream_viewer.receive_stream()


if __name__ == "__main__":
    main()
