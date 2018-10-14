import struct
import socket
import traceback
import math
import threading
import pyaudio
import numpy as np
import rtmidi
import time

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

from cv2 import *
import numpy as np
import rtmidi

levels = [0] * 5
# Setup for MIDI Ports
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)

if available_ports:
    midiout.open_port(0)
    print ("Port available")
else:
    midiout.open_virtual_port("My virtual output")



# Set up sentiment analysis
client = vision.ImageAnnotatorClient()
surprised = 1

#Sets up camera and video capture
cam = VideoCapture(0)   # 0 -> index of camera
start_time = time.time()
cam.read()
start_ms = cam.get(CAP_PROP_POS_MSEC)

joyful = False
surprised = False
sorrowful = False

def capture(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc.bind(('', port))
    try:
        while True:
            data = soc.recv(2000)
            if len(data) > 0:
                points = handle_packet(data)
                # 224 * math.pi / 180 <= p[0] <= 226 * math.pi / 180 and
                #targets = [0]
                targets = [72 * i for i in range(5)]
                targets = [(t - 1, t + 1) for t in targets]
                def get_fader(angle):
                    angle *= 180 / math.pi
                    for i, (start, end) in enumerate(targets):
                        start %= 360
                        end %= 360
                        if start > end and ((angle > start) or (angle < end)):
                            return i
                        if start <= angle <= end:
                            return i
                    return -1
                def get_value(pts):
                    if not pts:
                        return None
                    dists = [p[2] for p in pts]
                    dist = sum(dists) / len(dists)
                    min_dist = 0.6
                    max_dist = 2.8
                    return 1 - min(1, max(0, (dist - min_dist) / (max_dist - min_dist)))
                points = [p for p in points if get_fader(p[0]) != -1 and -math.pi / 180 <= p[1] <= math.pi / 180 and p[2] < 3]
                line_points = [[] for i in range(len(targets))]
                # .6 -> 0, 2.8 -> 1
                for p in points:
                    line_points[get_fader(p[0])].append(p)
                line_values = [get_value(lps) for lps in line_points]
                for i, value in enumerate(line_values):
                    if value != None:
                        levels[i] = value
                        print('Update: fader #{} = {}'.format(i, value))
                # colors = 'bgryc'
                # for color, lps in zip(colors, line_points):
                #     theta = [p[0] for p in lps]
                #     r = [p[2] for p in lps]
                #     plt.polar(theta, r, '{}.'.format(color))
                # plt.show(block = False)
                # plt.gcf().canvas.flush_events()
                # theta = [p[0] for p in points]
                # r = [p[2] for p in points]
                # if r:
                #     print(len(r), sum(r) / len(r))
                #plt.clf()
                #plt.polar(theta, r, 'r.')
                #plt.show(block=False)
                #plt.gcf().canvas.flush_events()
    except KeyboardInterrupt as e:
        return

def handle_packet(data):
    assert len(data) == 1206, len(data)
    timestamp, factory = struct.unpack_from("<IH", data, offset=1200)
    # Convert to seconds.
    timestamp /= 1e6
    #print('End:', timestamp, factory)
    points = []
    for offset in range(0, 1200, 100):
        points += handle_chunk(data[offset:offset+100])
    return points

ROTATION_MAX_UNITS = 36000
NUM_LASERS = 16
LASER_ANGLES = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]

def polar_to_cartesian(azimuth, altitude, dist):
    x = dist * math.cos(altitude) * math.sin(azimuth)
    y = dist * math.cos(altitude) * math.cos(azimuth)
    z = dist * math.sin(altitude)
    return (x, y, z)

def handle_chunk(chunk):
    points = []
    flag, azimuth = struct.unpack_from("<HH", chunk, 0)
    assert flag == 0xEEFF, hex(flag)
    # Convert to radians.
    azimuth = azimuth / 100 * math.pi / 180
    for step in range(2):
        # TODO interpolate
        #azimuth += step
        #azimuth %= 360
        # H-distance (2mm step), B-reflectivity
        arr = struct.unpack_from('<' + "HB" * 16, chunk, 4 + step * 48)
        for i in range(NUM_LASERS):
            dist = arr[i * 2]
            refl = arr[i * 2 + 1]
            # Convert to meters.
            dist = dist * 2 / 1000
            # Convert to range [0, 1]
            refl /= 255
            altitude = LASER_ANGLES[i] * math.pi / 180
            if dist > 0:
                points.append((azimuth, altitude, dist, refl))
    return points

def get_sentiments():
    for i in range(5):
        # Skip 5 frames.
        cam.read()
    
    s, img = cam.read()
    content = imencode('.png', img)[1].tostring()
    image = types.Image(content=content)
    # Performs label detection on the image file
    response = client.face_detection(image=image)
    for face in response.face_annotations:
        global joyful, surprised, sorrowful
        joyful = face.joy_likelihood > 1
        surprised = face.surprise_likelihood > 1
        sorrowful = face.sorrow_likelihood > 1

def sentiment_loop():
    while True:
        get_sentiments()

def midi_sentiment():
    while True:
        if joyful:
            cc_message1 = [0xB0, 26, 127] # Send continous control midi message to controller 1
            midiout.send_message(cc_message1) 
            cc_message2 = [0xB0, 27, 0] # Send continous control midi message to controller 2
            midiout.send_message(cc_message2) 

            print("JOY")
            time.sleep(1)

        if surprised:
            cc_message1 = [0xB0, 26, 0] # Send continous control midi message to controller 1
            midiout.send_message(cc_message1) 
            cc_message2 = [0xB0, 27, 127] # Send continous control midi message to controller 2
            midiout.send_message(cc_message2) 
            
            print("SUPRISE")
            time.sleep(1)
        
        time.sleep(0.1)


def midi_faders():
    while True:
        for i in range(5):
            cc_message = [0xB0, 21+i, levels[i]*127] 
            midiout.send_message(cc_message)
            time.sleep(0.01)
            print(cc_message)

if __name__ == '__main__':
    threading.Thread(target=lambda: capture(2368)).start()
    threading.Thread(target=midi_faders).start()
    threading.Thread(target=sentiment_loop).start()
    threading.Thread(target=midi_sentiment).start()