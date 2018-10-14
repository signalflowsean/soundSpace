import struct
import socket
import traceback
import math
import matplotlib
from matplotlib import pyplot as plt
import threading
import pyaudio
import numpy as np
import sklearn.cluster
import time

objlock = threading.Lock()
objects = []
#fig, ax = plt.subplots()
#plt.show(block=False)

def handle_points(points):
    # Reset switch values.
    points = [(*polar_to_cartesian(*p[:3]), *p) for p in points]
    points = [p for p in points if p[5] < 2.8 and -1 < p[2] < 1]
    # TODO try including reflectivity
    cart = [p[:3] for p in points]
    mcart = np.array([np.array(p) for p in cart])
    if not cart:
        return
    core_samples, labels = sklearn.cluster.dbscan(mcart, eps=0.2, min_samples=20)
    clusters = set(labels) - {-1}
    n_clusters = len(clusters)
    #ax.cla()
    #ax.set_xlim(-2.8, 2.8)
    #ax.set_ylim(-2.8, 2.8)
    global objects
    objlock.acquire(True)
    objects = []
    for i, label in enumerate(clusters):
        pts = mcart[labels == label]
        center = np.sum(pts, axis=0) / len(pts)
        #ax.plot([center[0]], [center[1]], 'bo')
        objects.append(center)
    objlock.release()
    x = [p[0] for p in cart]
    y = [p[1] for p in cart]
    hues = np.arange(n_clusters) / n_clusters
    colors = [matplotlib.colors.hsv_to_rgb((h, 1.0, 1.0)) for h in hues] + [(0, 0, 0)]
    #print(colors)
    #colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'orange', 'gray', 'cyan', 'purple', 'gray', 'indigo', 'lime', 'navy', 'fuschia', 'black']
    #ax.scatter(x, y, 1, c=[colors[label] for label in labels])
    # im = [[0] * 4 for _ in range(4)]
    # for i, (x, y, _, _) in enumerate(grid):
    #     im[y][x] = switches[i]
    # if switches != old_switches:
    #     ax.imshow(im, cmap=plt.cm.binary, origin='lower')
    #fig.canvas.draw()
    #fig.canvas.flush_events()

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
                if not new_points:
                    continue
                start_azimuth = new_points[0][0]
                end_azimuth = new_points[-1][0]
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
    azimuth = np.arctan2(y, x) % (2 * np.pi)
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

    chunk_size = 2048
    t = 0
    old_freq = 0
    mult = 0
    offset = 0
    while True:
        chunk = np.zeros(chunk_size)
        objlock.acquire(True)
        polars = [cartesian_to_polar(*center) for center in objects]
        objlock.release()
        polars.sort(key=lambda p: p[2])
        total = 0
        #for i, (az, alt, dist) in enumerate(polars[:1]):
        i = 0
        if not polars:
            stream.write(chunk.astype(np.float32).tostring())
            continue
        az, alt, dist = polars[0]
            #print('HELLO', i, az / (2 * np.pi), center)
        pc = az / (2 * np.pi) * 12
        freq = 440 * (2**(pc/12))
        shep = shepard_tones(pc)
        level = 1 - min(1, max(0, (dist - 0.5) / 2.3))
        amp = 0.1 * (np.exp(level * 2) - 1)
        total += amp
        print(i, pc, freq, amp)
        #print('F', pc, old_freq, freq)
        #freqs = np.linspace(old_freq, freq, chunk_size)
        old_t = t - chunk_size
        old_offset = offset
        old_mult = mult
        last_phase = t * old_mult + old_offset
        mult = freq / fs * 2 * np.pi
        offset = last_phase - t * mult
        for scale, freq_scale in [(pc / 12, 1 / 2), (1 - pc / 12, 1)]:#scale, freq in shep:
            last_phase = t * old_mult * freq_scale + old_offset * freq_scale
            tmp_offset = last_phase - t * mult * freq_scale
            chunk += scale / 2 * amp * np.sin((t + np.arange(chunk_size)) * mult * freq_scale + tmp_offset)
            print('edges', chunk[0], chunk[-1])
        #old_freq = freq
        print(total)
        if total > 1:
            chunk /= total
        stream.write(chunk.astype(np.float32).tostring())
        t += chunk_size

    stream.stop_stream()
    stream.close()

    p.terminate()

def shepard_tones(pc):
    mid_freq = 440 * (2**(pc/12))
    bot_freq = mid_freq / 2
    frac = pc / 12
    return ((frac, bot_freq), (1 - frac, mid_freq))

if __name__ == '__main__':
    threading.Thread(target=audio_loop).start()
    capture(2368)
    #threading.Thread(target=lambda: capture(2368)).start()