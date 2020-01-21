import os
import numpy as np
import scipy.io as sio
import plotly.graph_objs as go

""" Global variables """
joint_connections = ((0,1), (1,2), (2,3), (3,4),
			   		 (0,5), (5,6), (6,7), (7,8),
			   		 (0,9), (9,10), (10,11), (11,12),
			   		 (0,13), (13,14), (14,15), (15,16),
			   		 (0,17), (17,18), (18,19), (19,20))

color_blue = "rgb(0, 0, 255)"
color_red = "rgb(255, 0, 0)"

def get_trace3d(points3d, point_color=None, line_color=None, name="PointCloud"):
	""" Yields plotly traces for visualization """
	if point_color is None:
		point_color = "rgb(30, 20, 160)"
	if line_color is None:
		line_color = "rgb(30, 20, 160)"

	# Trace of points
	trace_pts = go.Scatter3d(x=points3d[:,0], y=points3d[:,2], z=points3d[:,1],
							 mode="markers", name=name,
							 marker=dict(symbol="circle", size=6, color=point_color))

	# Trace of line
	xlines, ylines, zlines = [],[],[]
	for line in joint_connections:
		for point in line:
			xlines.append(points3d[point, 0])
			ylines.append(points3d[point, 2])
			zlines.append(points3d[point, 1])
		xlines.append(None)
		ylines.append(None)
		zlines.append(None)
	trace_lines = go.Scatter3d(x=xlines, y=ylines, z=zlines,
							   mode="lines", name=name,
							   line=dict(width=6,color=line_color))

	return [trace_pts, trace_lines]

def get_figure3d(points3d, gt=None, range_scale=1):
	""" Yields plotly figure for visualization """
	traces = get_trace3d(points3d, color_blue, color_blue, "Predicted KP")
	if gt is not None:
		traces += get_trace3d(gt, color_red, color_red, "Groundtruth KP")

	layout = go.Layout(scene=dict(
								  aspectratio=dict(x=0.8, y=0.8, z=2),								  
								  xaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale)),
								  yaxis=dict(range=(-0.4 * range_scale, 0.4 * range_scale)),
								  zaxis=dict(range=(-1 * range_scale, 1 * range_scale))),
					   width=700,
					   margin=dict(r=20, l=10, b=10, t=10))
	return go.Figure(data=traces, layout=layout)
