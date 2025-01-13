import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import copy
import sys
from tqdm import tqdm
sys.path.append('..')
import deeplabv3plus.network as network
import argparse
import onnx
import onnxruntime as ort
#fix segmentation network

def get_calib_from_file(calib_file):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(calib_file, 'r') as f:
        lines = [line for line in f.readlines() if line.strip()]
    for line in lines:
        key, value = line.split(':', 1)
        # The only non-float values in these files are dates, which
        # we don't care about anyway
        try:
            if key == 'R0_rect':
                data['R0'] = torch.tensor([float(x) for x in value.split()]).reshape(3, 3)
            else:
                data[key] = torch.tensor([float(x) for x in value.split()]).reshape(3, 4)
        except ValueError:
            pass

    return data

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class Painter:
    def __init__(self, args, onnx=False):
        self.root_split_path = args.training_path
        self.save_path = os.path.join(args.training_path, "painted_lidar/")
        self.onnx = onnx
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.seg_net_index = 0
        self.model = None
        print(f'Using Segmentation Network -- deeplabv3plus')
        checkpoint_file = args.model_path
        if self.onnx:
            model = ort.InferenceSession(checkpoint_file)
            self.input_image_name = model.get_inputs()[0].name
        else:
            model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=19, output_stride=16)
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
        self.model = model
        self.cam_sync = args.cam_sync

        
    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne/' + ('%s.bin' % idx))
        return torch.from_numpy(np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 6))

    def get_image(self, idx, camera):
        filename = os.path.join(self.root_split_path, camera + ('%s.jpg' % idx))
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        return input_batch
    
    def get_model_output(self, input_batch):
        if self.onnx:
            input_data = {self.input_image_name: to_numpy(input_batch)}
            output = self.model.run(None, input_data)
            output = [torch.from_numpy(item) for item in output]
        else:
            with torch.no_grad():
                output = self.model(input_batch)[0]
        return output

    def get_score(self, model_output):
        sf = torch.nn.Softmax(dim=2)
        output_permute = model_output.permute(1,2,0)
        output_permute = sf(output_permute)
        output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 6).to(device=model_output.device)
        output_reassign[:,:,0] = torch.sum(output_permute[:,:,:11], dim=2) # background
        output_reassign[:,:,1] = output_permute[:,:,18] #bicycle
        output_reassign[:,:,2] = torch.sum(output_permute[:,:,[13, 14, 15, 16]], dim=2) # vehicles
        output_reassign[:,:,3] = output_permute[:,:,11] #person
        output_reassign[:,:,4] = output_permute[:,:,12] #rider
        output_reassign[:,:,5] = output_permute[:,:,17] # motorcycle

        return output_reassign
    
    def get_calib_fromfile(self, idx, device):
        calib_file = os.path.join(self.root_split_path, 'calib/' + ('%s.txt' % idx))
        calib = get_calib_from_file(calib_file)
        calib['P0'] = torch.cat([calib['P0'], torch.tensor([[0., 0., 0., 1.]])], axis=0).to(device=device)
        calib['P1'] = torch.cat([calib['P1'], torch.tensor([[0., 0., 0., 1.]])], axis=0).to(device=device)
        calib['P2'] = torch.cat([calib['P2'], torch.tensor([[0., 0., 0., 1.]])], axis=0).to(device=device)
        calib['P3'] = torch.cat([calib['P3'], torch.tensor([[0., 0., 0., 1.]])], axis=0).to(device=device)
        calib['P4'] = torch.cat([calib['P4'], torch.tensor([[0., 0., 0., 1.]])], axis=0).to(device=device)
        calib['R0_rect'] = torch.zeros([4, 4], dtype=calib['R0'].dtype, device=device)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0'].to(device=device)
        calib['Tr_velo_to_cam_0'] = torch.cat([calib['Tr_velo_to_cam_0'], torch.tensor([[0., 0., 0., 1.]], )], axis=0).to(device=device)
        calib['Tr_velo_to_cam_1'] = torch.cat([calib['Tr_velo_to_cam_1'], torch.tensor([[0., 0., 0., 1.]], )], axis=0).to(device=device)
        calib['Tr_velo_to_cam_2'] = torch.cat([calib['Tr_velo_to_cam_2'], torch.tensor([[0., 0., 0., 1.]], )], axis=0).to(device=device)
        calib['Tr_velo_to_cam_3'] = torch.cat([calib['Tr_velo_to_cam_3'], torch.tensor([[0., 0., 0., 1.]], )], axis=0).to(device=device)
        calib['Tr_velo_to_cam_4'] = torch.cat([calib['Tr_velo_to_cam_4'], torch.tensor([[0., 0., 0., 1.]], )], axis=0).to(device=device)
        return calib
    
    def cam_to_lidar(self, pointcloud, projection_mats, camera_num):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo_to_cam_' + str(camera_num)].matmul(lidar_velo_coords.transpose(0, 1))
        lidar_cam_coords = lidar_cam_coords.transpose(0, 1)
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def project_points_mask(self, lidar_cam_points, projection_mats, class_scores, camera_num):
        points_projected_on_mask = projection_mats['P' + str(camera_num)].matmul(projection_mats['R0_rect'].matmul(lidar_cam_points.transpose(0, 1)))
        points_projected_on_mask = points_projected_on_mask.transpose(0, 1)
        points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores[camera_num].shape[1]) #x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores[camera_num].shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img & (lidar_cam_points[:, 2] > 0)

        points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
        points_projected_on_mask = torch.floor(points_projected_on_mask).int() # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask = points_projected_on_mask[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array
        return (points_projected_on_mask, true_where_point_on_img)

    def augment_lidar_class_scores_both(self, class_scores, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        """
        #lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats? 
        ################################
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 0)

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_0, true_where_point_on_img_0 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 0)
        
        # left
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 1)
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_1, true_where_point_on_img_1 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 1)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 2)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_2, true_where_point_on_img_2 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 2)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 3)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_3, true_where_point_on_img_3 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 3)
        
        lidar_cam_coords = self.cam_to_lidar(lidar_raw[:,:4], projection_mats, 4)
        lidar_cam_coords[:, -1] = 1
        points_projected_on_mask_4, true_where_point_on_img_4 = self.project_points_mask(lidar_cam_coords, projection_mats, class_scores, 4)

        true_where_point_on_both_0_1 = true_where_point_on_img_0 & true_where_point_on_img_1
        true_where_point_on_both_0_2 = true_where_point_on_img_0 & true_where_point_on_img_2
        true_where_point_on_both_1_3 = true_where_point_on_img_1 & true_where_point_on_img_3
        true_where_point_on_both_2_4 = true_where_point_on_img_2 & true_where_point_on_img_4
        true_where_point_on_img = true_where_point_on_img_1 | true_where_point_on_img_0 | true_where_point_on_img_2 | true_where_point_on_img_3 | true_where_point_on_img_4

        point_scores_0 = class_scores[0][points_projected_on_mask_0[:, 1], points_projected_on_mask_0[:, 0]].reshape(-1, class_scores[0].shape[2])
        point_scores_1 = class_scores[1][points_projected_on_mask_1[:, 1], points_projected_on_mask_1[:, 0]].reshape(-1, class_scores[1].shape[2])
        point_scores_2 = class_scores[2][points_projected_on_mask_2[:, 1], points_projected_on_mask_2[:, 0]].reshape(-1, class_scores[2].shape[2])
        point_scores_3 = class_scores[3][points_projected_on_mask_3[:, 1], points_projected_on_mask_3[:, 0]].reshape(-1, class_scores[3].shape[2])
        point_scores_4 = class_scores[4][points_projected_on_mask_4[:, 1], points_projected_on_mask_4[:, 0]].reshape(-1, class_scores[4].shape[2])

        augmented_lidar = torch.cat((lidar_raw[:,:5], torch.zeros((lidar_raw.shape[0], class_scores[1].shape[2])).to(device=lidar_raw.device)), axis=1)
        augmented_lidar[true_where_point_on_img_0, -class_scores[0].shape[2]:] += point_scores_0
        augmented_lidar[true_where_point_on_img_1, -class_scores[1].shape[2]:] += point_scores_1
        augmented_lidar[true_where_point_on_img_2, -class_scores[2].shape[2]:] += point_scores_2
        augmented_lidar[true_where_point_on_img_3, -class_scores[3].shape[2]:] += point_scores_3
        augmented_lidar[true_where_point_on_img_4, -class_scores[4].shape[2]:] += point_scores_4
        augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:]
        augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:]
        if self.cam_sync:
            augmented_lidar = augmented_lidar[true_where_point_on_img]

        return augmented_lidar

    def run(self):
        image_files = os.listdir(os.path.join(self.root_split_path, 'image_0'))
        for img_file in tqdm(image_files):
            
            sample_idx = os.path.splitext(img_file)[0]
            # points: N * 4(x, y, z, r)
            points = self.get_lidar(sample_idx)
            
            # get segmentation score from network
            scores_from_cam = []
            for i in range(5):
                input_batch = self.get_image(sample_idx, 'image_' + str(i) + '/')
                output = self.get_model_output(input_batch)
                scores_from_cam.append(self.get_score(output))
            # scores_from_cam: H * W * 4/5, each pixel have 4/5 scores(0: background, 1: bicycle, 2: car, 3: person, 4: rider)
            device = scores_from_cam[0].device
            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx, device)
            points = points.to(device=device)
            
            # paint the point clouds
            # points: N * 8
            points = self.augment_lidar_class_scores_both(scores_from_cam, points, calib_fromfile).cpu()
            
            np.save(self.save_path + ("%s.npy" % sample_idx), points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--training_path', help='your data root for the training data', required=True)
    parser.add_argument('--model_path', help='path to segmentation model', required=True)
    parser.add_argument('--cam_sync', action='store_true', help='only use objects visible to a camera')
    args = parser.parse_args()
    painter = Painter(args)
    painter.run()