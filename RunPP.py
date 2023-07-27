import argparse
import cv2
import numpy as np
import os
import torch
import pdb
import cv2
import open3d as o3d
import os

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillars
from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars
from loss import Loss
from torch.utils.tensorboard import SummaryWriter
from bbox_utils.point_cloud import PointCloud
from bbox_utils import BoundingBox3D
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# def read_points(file_path, dim=4):
#     suffix = os.path.splitext(file_path)[1]
#     assert suffix in ['.bin', '.ply']
#     if suffix == '.bin':
#         return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
#     else:
#         raise NotImplementedError

# import glob
# file_list = glob.glob(os.path.join(test_path, '*.bin'))

# def addBB(data,resv,reslabels,scores):
#   colorA=['rgb(127, 127, 127)','rgb(170, 50, 20)','rgb(30, 127, 30)']
#   for i,res in enumerate(resv):
#     values = {"position": {"x": res[0],"y": res[1],"z": res[2]-0.3},"rotation": {"x": 0,"y": 0,"z": -res[6]},"dimensions": {"x": res[3],"y": res[4],"z": res[5]}}
#     rotation = np.array([values["rotation"]["x"], values["rotation"]["y"], values["rotation"]["z"]])
#     bbox = BoundingBox3D(
#     x=values["position"]["x"],
#     y=values["position"]["y"],
#     z=values["position"]["z"],
#     length=values["dimensions"]["x"],
#     width=values["dimensions"]["y"],
#     height=values["dimensions"]["z"],
#     euler_angles=rotation,)
#     corners, triangle_vertices = bbox.p, bbox.triangle_vertices
#     data.append(go.Mesh3d(
#         x=corners[:, 0],
#         y=corners[:, 1],
#         z=corners[:, 2],
#         i=triangle_vertices[0],
#         j=triangle_vertices[1],
#         k=triangle_vertices[2],
#         opacity=0.6,
#         color=colorA[reslabels[i]],
#         flatshading=True,text='confindence: {:.2%}'.format(scores[i])))
#
#   return data

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts


def main(args):
    # CLASSES = {
    #     'Pedestrian': 0,
    #     'Cyclist': 1,
    #     'Car': 2
    # }
    CLASSES = {
        'Trunk': 0
    }
    model = PointPillars(nclasses=len(CLASSES)).cuda()
    model.load_state_dict(torch.load(args.ckpt))
    file_name=args.pc_path
    pc=args.array
   # pc = read_points(args.array)
    pc_torch = torch.from_numpy(pc)
    pc_torch = pc_torch.cuda()
    result_filter = model(batched_pts=[pc_torch], mode='test')[0]

    return result_filter
    # print(result_filter)
    # try:
    #     resbb = result_filter['lidar_bboxes']
    #     reslabels = result_filter['labels']
    #     scores = result_filter['scores']
    #     lidar_bboxes = result_filter['lidar_bboxes']
    #     labels, scores = result_filter['labels'], result_filter['scores']
    #     print("step 1")
    #     pc = pc.T
    #     x = pc[0, :];
    #     point_size = np.full(x.shape, 2)
    #     data = []
    #     x = pc[0, :];
    #     point_size = np.full(x.shape, 2)
    #     fig = px.scatter_3d(x=pc[0, :], y=pc[1, :], z=pc[2, :], size=point_size, opacity=1)
    #     # data = [go.Scatter3d(x=pc[0,:], y=pc[1,:], z=pc[2,:], mode="markers", marker=dict(size=2))]
    #     data = [go.Scatter3d(x=pc[0, :], y=pc[1, :], z=pc[2, :], mode="markers",
    #                          marker=dict(size=2, color=pc[2, :], colorscale='Viridis'))]
    #
    #     mega_centroid = np.average(pc, axis=1)
    #     mega_max = np.amax(pc, axis=1)
    #     mega_min = np.amin(pc, axis=1)
    #     lower_bound = mega_centroid - 40  # (np.amax(mega_max - mega_min) / 2)
    #     upper_bound = mega_centroid + 40  # (np.amax(mega_max - mega_min) / 2)
    #     data = addBB(data, resbb, reslabels, scores)
    #     show_grid_lines = True
    #     # Setup layout
    #     grid_lines_color = 'rgb(127, 127, 127)' if show_grid_lines else 'rgb(30, 30, 30)'
    #     layout = go.Layout(scene=dict(
    #         xaxis=dict(nticks=8,
    #                    range=[lower_bound[0], upper_bound[0]],
    #                    showbackground=True,
    #                    backgroundcolor='rgb(30, 30, 30)',
    #                    gridcolor=grid_lines_color,
    #                    zerolinecolor=grid_lines_color),
    #         yaxis=dict(nticks=8,
    #                    range=[lower_bound[1], upper_bound[1]],
    #                    showbackground=True,
    #                    backgroundcolor='rgb(30, 30, 30)',
    #                    gridcolor=grid_lines_color,
    #                    zerolinecolor=grid_lines_color),
    #         zaxis=dict(nticks=8,
    #                    range=[lower_bound[2], upper_bound[2]],
    #                    showbackground=True,
    #                    backgroundcolor='rgb(30, 30, 30)',
    #                    gridcolor=grid_lines_color,
    #                    zerolinecolor=grid_lines_color),
    #         xaxis_title="x (meters)",
    #         yaxis_title="y (meters)",
    #         zaxis_title="z (meters)"
    #     ),
    #         margin=dict(r=10, l=10, b=10, t=10),
    #         paper_bgcolor='rgb(30, 30, 30)',
    #         font=dict(
    #             family="Courier New, monospace",
    #             color=grid_lines_color
    #         ),
    #         legend=dict(
    #             font=dict(
    #                 family="Courier New, monospace",
    #                 color='rgb(127, 127, 127)'
    #             )
    #         )
    #     )
    #     fig = go.Figure(data=data, layout=layout)
    #     # fig.show()
    #     filename = os.path.basename(file_name)
    #     html_path = os.path.join(args.res_path, filename[:-3] + 'html')
    #     fig.write_html(html_path)
    #     print("file save: {}".format(html_path))
    # except:
    #     print("An exception occurred")

    # LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    # pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Configuration Parameters')
    # parser.add_argument('--ckpt', default='/home/roy/Projects/trunk_detector/pretrained/epoch_100.pth', help='your checkpoint for kitti')
    # parser.add_argument('--pc_path', default='/home/roy/Projects/trunk_detector/tests/008024_52804.bin', help='your point cloud path')
    # # parser.add_argument('--no_cuda', action='store_true',
    # #                     help='whether to use cuda')
    # parser.add_argument('--res_path', default='/home/roy/Projects/trunk_detector/html_res', help='your output html path')
    # parser.add_argument('--my_array', nargs='+', type=float, default=[1, 2, 3, 4], help='Your input array')
    #
    # args = parser.parse_args()
    outputs=main(args)
