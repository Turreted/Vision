import argparse
from time import sleep
from time import time

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import threading

class StreamViewer:
    def __init__(self, port=PORT):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + port)
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode_(''))
        self.current_frame = None
        self.keep_running = True

        context = zmq.Context()
        self.resp_socket = context.socket(zmq.PUB)
        self.resp_socket.connect('tcp://' + 'raspberrypi.local' + ':' + str(int(port)+1)) # TODO: might cause bugs
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

        while self.resp_socket and self.keep_running:

        # Do openCV image processing here
            try:
                self.resp_socket.send_string("dummy data")
            except KeyboardInterrupt:
                self.keep_running = False
                break
        
            sleep(0.5)
        
                
    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False

def main():
    port = PORT

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Viewer to use, default'
                             ' is ' + PORT, required=False)

    args = parser.parse_args()
    if args.port:
        port = args.port

    stream_viewer = StreamViewer(port)
    stream_viewer.receive_stream()


if __name__ == '__main__':
    main()
