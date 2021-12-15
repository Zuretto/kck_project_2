from IPython.display import YouTubeVideo
from IPython.display import Video
from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as stats
from scipy.io import wavfile


samplerate, data = wavfile.read('train/001_K.wav')

