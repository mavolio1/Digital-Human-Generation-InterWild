import torch
import numpy as np
import json
from glob import glob
import os.path as osp
import smplx
import cv2
import os
import random
from plyfile import PlyData, PlyElement
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
DirectionalLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)

def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x,y,z),1)
    return v

def render_mesh(mesh, face, cam_param, render_shape, hand_type):
    mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3)

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cpu())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cpu().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cpu()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    if hand_type == 'right':
        color = ((1.0, 0.0, 0.0),)
    else:
        color = ((0.0, 1.0, 0.0),)
    materials = Materials(
	device='cuda',
        specular_color=color,
	shininess=0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
    
    image = images[:,:,:,:3] * 255
    depthmap = fragments.zbuf.float()
    mask = depthmap > 0
    return image, depthmap, mask

data_root_path = ''
mano_root_path = ''
capture_id = 'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands'
envmap_mode = 'envmap_per_frame' # envmap_per_frame, envmap_per_segment

mano_layer = {'right': smplx.create(mano_root_path, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False).cpu(), 'left': smplx.create(mano_root_path, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False).cpu()}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

with open(osp.join(data_root_path, capture_id, 'Mugsy_cameras', 'cam_params.json')) as f:
    cam_params = json.load(f)

cam_path_list = glob(osp.join(data_root_path, capture_id, 'Mugsy_cameras', envmap_mode, 'images', '*'))
for cam_path in cam_path_list:
    cam_name = cam_path.split('/')[-1]
    cam_param = cam_params[cam_name]
    R, t = torch.FloatTensor(cam_param['R']).view(1,3,3).cpu(), torch.FloatTensor(cam_param['t']).view(1,3).cpu() / 1000 # millimeter to meter
    focal, princpt = torch.FloatTensor(cam_param['focal']).view(1,2).cpu(), torch.FloatTensor(cam_param['princpt']).view(1,2).cpu()

    img_path_list = glob(osp.join(cam_path, '*.png'))
    for img_path in img_path_list:
        frame_idx = int(img_path.split('/')[-1][:-4])
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        prev_depth = None
        for h in ('right', 'left'):
            # option 1: get mesh from parameters
            with open(osp.join(data_root_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_' + h + '.json')) as f:
                mano_param = json.load(f)
            pose = torch.FloatTensor(mano_param['pose']).view(1,-1).cpu()
            shape = torch.FloatTensor(mano_param['shape']).view(1,-1).cpu()
            trans = torch.FloatTensor(mano_param['trans']).view(1,-1).cpu()
            with torch.no_grad():
                output = mano_layer[h](global_orient=pose[:,:3], hand_pose=pose[:,3:], betas=shape, transl=trans)
            mesh = output.vertices
            
            # option 2: load mesh
            #mesh = load_ply(osp.join(data_root_path, capture_id, 'mano_fits', 'meshes', str(frame_idx) + '_' + h + '.ply'))
            #mesh = mesh / 1000 # millimeter to meter
            #mesh = torch.from_numpy(mesh).float().cpu()[None,:,:]

            # render
            face = torch.LongTensor(mano_layer[h].faces.astype(np.int32))[None,:,:].cpu()
            rgb, depth, valid_mask = render_mesh(mesh, face, {'R': R, 't': t, 'focal': focal, 'princpt': princpt}, (img_height, img_width), h)
            rgb, depth, valid_mask = rgb.cpu().numpy()[0], depth.cpu().numpy()[0], valid_mask.cpu().numpy()[0]
            if prev_depth is None:
                render_mask = valid_mask
                img = rgb * render_mask + img * (1 - render_mask)
                prev_depth = depth
            else:
                render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth<=0)
                img = rgb * render_mask + img * (1 - render_mask)
                prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

        cv2.imwrite(capture_id + '_' + cam_name + '_' + str(frame_idx) + '.jpg', img)
        print('saved')




