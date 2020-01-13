import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import copy
import plotly.graph_objs as go
import plotly.offline as py

BLUE = "rgb(90, 130, 238)"
RED = "rgb(205, 90, 76)"

CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (18, 19), (19, 20))

def get_trace3d(points3d, point_color=None, line_color=None, name="PointCloud"):
    """Yields plotly traces for visualization."""
    if point_color is None:
        point_color = "rgb(30, 20, 160)"
    if line_color is None:
        line_color = "rgb(30, 20, 160)"
    # Trace of points.
    trace_of_points = go.Scatter3d(
        x=points3d[:, 0],
        y=points3d[:, 2],
        z=points3d[:, 1],
        mode="markers",
        name=name,
        marker=dict(
            symbol="circle",
            size=3,
            color=point_color))

    # Trace of lines.
    xlines = []
    ylines = []
    zlines = []
    for line in CONNECTIONS:
        for point in line:
            xlines.append(points3d[point, 0])
            ylines.append(points3d[point, 2])
            zlines.append(points3d[point, 1])
        xlines.append(None)
        ylines.append(None)
        zlines.append(None)
    trace_of_lines = go.Scatter3d(
        x=xlines,
        y=ylines,
        z=zlines,
        mode="lines",
        name=name,
        line=dict(color=line_color))
    return [trace_of_points, trace_of_lines]


def get_figure3d(points3d, gt=None, range_scale=1):
    """Yields plotly fig for visualization"""
    traces = get_trace3d(points3d, BLUE, BLUE, "prediction")
    if gt is not None:
        traces += get_trace3d(gt, RED, RED, "groundtruth")
    layout = go.Layout(
        scene=dict(
            aspectratio=dict(x=0.8,
                             y=0.8,
                             z=2),
            xaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale),),
            yaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale),),
            zaxis=dict(range=(-1 * range_scale, 1 * range_scale),),),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10))
    return go.Figure(data=traces, layout=layout)

def substract_mean(xyz):
    denom = (xyz[:,0].std() + xyz[:,1].std() + xyz[:,2].std()) / 3
    xyz_centered = np.zeros(xyz.shape)
    x = xyz[:,0] - np.sum(xyz[:,0])/len(xyz[:,0])
    y = xyz[:,1] - np.sum(xyz[:,1])/len(xyz[:,1])
    z = xyz[:,2] - np.sum(xyz[:,2])/len(xyz[:,2])
    xyz_centered[:,0] = x
    xyz_centered[:,1] = y
    xyz_centered[:,2] = z
    #xyz_centered /= denom
    return xyz_centered

pred = np.load("pred.npy")
pred_proc = np.load("pred_procr.npy")
ground_truth_test = np.load("ground_truth_test.npy")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
#while(i<len(ground_truth_test)):
gt = substract_mean(ground_truth_test[0])
pred = substract_mean(pred_proc[0])
#===========================================================================
# xg = gt[:,0]
# yg = gt[:,1]
# zg = gt[:,2]
# x = pred[:,0]
# y = pred[:,1]
# z = pred[:,2]
# i+=3
# #print(xg)
# #print(yg)
# min = np.min(gt)
# max = np.max(gt)
# ax.set_zlim(np.min(gt[:,0])-0.1,np.max(gt[:,0])+0.1)
# ax.set_xlim(np.min(gt[:,1])-0.1,np.max(gt[:,1])+0.1)
# ax.set_ylim(np.min(gt[:,2])-0.1,np.max(gt[:,2])+0.1)
# ax.scatter(xg, yg, zg, c='b', marker='o')
# ax.scatter(x, y, z, c='r', marker='o')
#===========================================================================
#plt.scatter(yg, xg,  marker='o')
#plt.scatter(y, x,  marker='+')
fig = get_figure3d(gt, pred)
py.plot(fig)
#plt.pause(5)
#plt.cla()