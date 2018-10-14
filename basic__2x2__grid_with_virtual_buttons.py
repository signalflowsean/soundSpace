import struct
import socket
import traceback
import math
import matplotlib
from matplotlib import pyplot as plt
import threading
import pyaudio
import numpy as np

switches = [0] * 4

def handle_points(points):
    points = [(*polar_to_cartesian(*p[:3]), *p) for p in points]
    points = [p for p in points if p[2] < 1 and p[5] < 5]
    grid = [(0, 0, -2, -2), (0, 0, -2, 2), (0, 0, 2, -2), (0, 0, 2, 2)]
    grid_points = [[] for _ in range(len(grid))]
    for p in points:
        for i, (x1, y1, x2, y2) in enumerate(grid):
            if (x1 <= p[0] <= x2 or x2 <= p[0] <= x1) and (y1 <= p[1] <= y2 or y2 <= p[1] <= y1):
                grid_points[i].append(p)
    for i, gp in enumerate(grid_points):
        if len(gp):
            max_z = max(p[2] for p in gp)
            z_thresh_high = 0.3
            z_thresh_low = 0.25
            if max_z >= z_thresh_high and not switches[i]:
                switches[i] = True
                print('switch {} is on'.format(i))
            elif max_z < z_thresh_low and switches[i]:
                switches[i] = False
                print('switch {} is off'.format(i))

def capture(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc.bind(('', port))
    try:
        last_azimuth = 2*np.pi
        points = []
        while True:
            data = soc.recv(2000)
            if len(data) > 0:
                new_points = handle_packet(data)
                start_azimuth = new_points[0][0]
                end_azimuth = new_points[-1][0]
                #min_azimuth = min(p[0] for p in new_points)
                #max_azimuth = max(p[0] for p in new_points)
                points += new_points
                if end_azimuth < last_azimuth:
                    handle_points(points)
                    points = []
                last_azimuth = end_azimuth
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

def cartesian_to_polar(x, y, z):
    dist = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    altitude = np.arccos(z / dist)
    return (azimuth, altitude, dist)

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

def audio_loop():
    p = pyaudio.PyAudio()

    fs = 22050
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    chunk_size = 1024
    t = 0
    while True:
        chunk = np.zeros(chunk_size)
        freqs = [440, 523.251, 554.365, 587.330, 659.255]
        for freq, level in zip(freqs, levels):
            mult = freq / fs * 2*np.pi
            chunk += 0.1 * (np.exp(level * 2) - 1) * np.sin((t + np.arange(chunk_size)) * mult)
        stream.write(chunk.astype(np.float32).tostring())
        t += chunk_size

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == '__main__':
    capture(2368)