import matplotlib.pyplot as plt
import numpy as np
import os
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))

def draw_wp(file_name='wp.csv'):
    x = []
    y = []

    with open(os.path.join(script_dir, file_name)) as f:
        waypoints_file = csv.reader(f)
        for i, row in enumerate(waypoints_file):
            x.append(float(row[0]))
            y.append(float(row[1]))

    plt.plot(x[:-400], y[:-400], '-o', markersize=0.01)
    plt.show()

draw_wp()
