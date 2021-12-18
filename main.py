from scipy import signal as sg
from scipy import fft
import numpy as np
from scipy.io import wavfile
import scipy.io.wavfile
import sys
import warnings

warnings.filterwarnings("ignore")


def main(file):
    w, signal = scipy.io.wavfile.read(file)
    window = sg.windows.blackman(len(signal)) #mozna sprawdzac dla roznych okien (blackmanharris, bohman i inne)
    if signal.ndim == 1:
        #sygnal jest jednokanalowy
        signal = np.array(signal) * window
        fouriered = abs(fft.fft(signal))
    else:
        #sygnal jest dwukanalowy - licze FFT ze sredniej lewego i prawego kanalu.
        signal = np.array(signal) * np.array([[i, i] for i in window])
        signal_l = np.array([i[0] for i in signal])
        signal_r = np.array([i[1] for i in signal])
        signal = (signal_l + signal_r) / 2
        fouriered = abs(fft.fft((signal_l + signal_r) / 2))

    #OX
    freqs = fft.fftfreq(len(signal), 1/w)

    combined = sorted(np.vstack((freqs, fouriered)).T, key=lambda x: x[0])
    combined = list(filter(lambda x: 0 <= x[0] <= 3000, combined))

    bests = []
    for idx in range(len(combined)):

        if 85 <= combined[idx][0] <= 255:
            bests.append([combined[idx], combined[idx][1]
                          * combined[idx * 2][1]
                          * combined[idx * 3][1]
                          * combined[idx * 4][1]])

    freq = [x[0] for x in [[x[0][0], x[1]] for x in bests]]
    y = [x[1] / 1e25 for x in [[x[0][0], x[1]] for x in bests]]

    if np.average(freq, weights=y) <= 167:
        print("M")
    else:
        print("K")


if __name__ == '__main__':
    main(sys.argv[1])
