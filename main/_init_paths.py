import os.path as osp
import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

#faster_rcnn_dir = "/home/mqadri/faster_rcnn.pytorch"
faster_rcnn_dir = "/home/mqadri/hand-integral-pose-estimation"
# Add lib to PYTHONPATH
#lib_path = osp.join(faster_rcnn_dir)
lib_path = osp.join(faster_rcnn_dir, 'lib')
add_path(lib_path)
print(sys.path)
#model_path = osp.join(faster_rcnn_dir, 'lib')
#add_path(model_path)

