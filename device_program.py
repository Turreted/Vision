import argparse
from time import sleep
from time import time

import cv2
import zmq
import numpy as np

from camera.Camera import Camera
from constants import PORT, SERVER_ADDRESS
from utils import image_to_string

import threading


class Streamer:
    def __init__(self, server_address=SERVER_ADDRESS, port=PORT):
        """
        Tries to connect to the StreamViewer with supplied server_address and creates a socket for future use.

        :param server_address: Address of the computer on which the StreamViewer is running, default is `localhost`
        :param port: Port which will be used for sending the stream
        """

        # bind sending socket
        print("Connecting to ", server_address, "at", port)
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.connect("tcp://" + server_address + ":" + port)
        self.keep_running = True

        # bind receiving socket
        context = zmq.Context()
        self.resp_socket = context.socket(zmq.SUB)
        self.resp_socket.bind("tcp://*:" + str(int(port) + 1))
        self.resp_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode_(""))

        self.person_recog = False
        x = threading.Thread(target=self.resp_thread, args=())
        x.start()

        self.stopwatch = time()

    def start(self):
        """
        Starts sending the stream to the Viewer.
        Creates a camera, takes a image frame converts the frame to string and sends the string across the network
        :return: None
        """
        print("Streaming Started...")
        camera = Camera()
        camera.start_capture()
        self.keep_running = True

        while self.footage_socket and self.keep_running:
            if time() - self.stopwatch > 0.07:
                try:
                    frame = camera.current_frame.read()  # grab the current frame
                    compressed = cv2.resize(
                        frame, (780, 540), interpolation=cv2.INTER_LINEAR
                    )
                    image_as_string = image_to_string(compressed)
                    self.footage_socket.send(image_as_string)

                except KeyboardInterrupt:
                    cv2.destroyAllWindows()
                    break

                self.stopwatch = time()

        print("Streaming Stopped!")
        cv2.destroyAllWindows()

    def resp_thread(self):
        while self.resp_socket and self.keep_running:
            try:
                data = self.resp_socket.recv_string(flags=zmq.NOBLOCK)
                if data:
                    print(data)
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    pass
                else:
                    exit()

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False


def main():
    port = PORT
    server_address = SERVER_ADDRESS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--server",
        help="IP Address of the server which you want to connect to, default"
        " is " + SERVER_ADDRESS,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--port",
        help="The port which you want the Streaming Server to use, default"
        " is " + PORT,
        required=False,
    )

    args = parser.parse_args()

    if args.port:
        port = args.port
    if args.server:
        server_address = args.server

    streamer = Streamer(server_address, port)
    streamer.start()


if __name__ == "__main__":
    main()
