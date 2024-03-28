import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import calibration_waymo
import copy
import sys
from tqdm import tqdm
sys.path.append('..')
import deeplabv3plus.network as network
import argparse
#fix segmentation network


class Painter:
    def __init__(self, args):
        self.root_split_path = args.training_path
        self.save_path = os.path.join(args.training_path, "painted_lidar/")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.seg_net_index = 0
        self.model = None
        print(f'Using Segmentation Network -- deeplabv3plus')
        checkpoint_file = args.model_path
        model = network.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=19, output_stride=16)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = torch.nn.DataParallel(model)
        #model.load_state_dict(torch.load(checkpoint_file))
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        self.model = model

        
    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 6)

    def get_score(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera 
        return:
            a tensor H  * W * 4(deeplab)/5(deeplabv3plus), for each pixel we have 4/5 scorer that sums to 1
        '''
        output_reassign_softmax = None
        if self.seg_net_index == 0:
            filename = self.root_split_path + left + ('%s.jpg' % idx)
            input_image = Image.open(filename)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')

            with torch.no_grad():
                output = self.model(input_batch)[0]

            output_permute = output.permute(1,2,0)
            output_probability,output_predictions =  output_permute.max(2)

            other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | (output_predictions == 7) | (output_predictions == 15))
            detect_object_mask = ~other_object_mask
            sf = torch.nn.Softmax(dim=2)

            # bicycle = 2 car = 7 person = 15 background = 0
            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1),4)
            output_reassign[:,:,0] = detect_object_mask * output_permute[:,:,0] + other_object_mask * output_probability # background
            output_reassign[:,:,1] = output_permute[:,:,2] # bicycle
            output_reassign[:,:,2] = output_permute[:,:,7] # car
            output_reassign[:,:,3] = output_permute[:,:,15] #person
            output_reassign_softmax = sf(output_reassign).cpu().numpy()

        elif self.seg_net_index == 1:
            filename = self.root_split_path + left + ('%s.png' % idx)
            result = inference_segmentor(self.model, filename)
            # person 11, rider 12, vehicle 13/14/15/16, bike 17/18
            output_permute = torch.tensor(result[0]).permute(1,2,0) # H, W, 19
            sf = torch.nn.Softmax(dim=2)

            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 5)
            output_reassign[:,:,0], _ = torch.max(output_permute[:,:,:11], dim=2) # background
            output_reassign[:,:,1], _ = torch.max(output_permute[:,:,[17, 18]], dim=2) # bicycle
            output_reassign[:,:,2], _ = torch.max(output_permute[:,:,[13, 14, 15, 16]], dim=2) # car
            output_reassign[:,:,3] = output_permute[:,:,11] #person
            output_reassign[:,:,4] = output_permute[:,:,12] #rider
            output_reassign_softmax = sf(output_reassign).cpu().numpy()
        
        elif self.seg_net_index == 2:
            filename = self.root_split_path + "score_hma/" + left + ('%s.npy' % idx)
            output_reassign_softmax = np.load(filename)

        return output_reassign_softmax
    
    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = calibration_waymo.get_calib_from_file(calib_file)
        calib['P0'] = np.concatenate([calib['P0'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P1'] = np.concatenate([calib['P1'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P4'] = np.concatenate([calib['P4'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo_to_cam_0'] = np.concatenate([calib['Tr_velo_to_cam_0'], np.array([[0., 0., 0., 1.]], )], axis=0)
        calib['Tr_velo_to_cam_1'] = np.concatenate([calib['Tr_velo_to_cam_1'], np.array([[0., 0., 0., 1.]], )], axis=0)
        calib['Tr_velo_to_cam_2'] = np.concatenate([calib['Tr_velo_to_cam_2'], np.array([[0., 0., 0., 1.]], )], axis=0)
        calib['Tr_velo_to_cam_3'] = np.concatenate([calib['Tr_velo_to_cam_3'], np.array([[0., 0., 0., 1.]], )], axis=0)
        calib['Tr_velo_to_cam_4'] = np.concatenate([calib['Tr_velo_to_cam_4'], np.array([[0., 0., 0., 1.]], )], axis=0)
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
        lidar_cam_coords = projection_mats['Tr_velo_to_cam_' + str(camera_num)].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords

    def project_points_mask(self, lidar_cam_points, projection_mats, class_scores, camera_num):
        points_projected_on_mask = projection_mats['P' + str(camera_num)].dot(projection_mats['R0_rect'].dot(lidar_cam_points.transpose()))
        points_projected_on_mask = points_projected_on_mask.transpose()
        points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

        true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores[camera_num].shape[1]) #x in img coords is cols of img
        true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores[camera_num].shape[0])
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

        points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
        points_projected_on_mask = np.floor(points_projected_on_mask).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
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

        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        #socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_0 = class_scores[0][points_projected_on_mask_0[:, 1], points_projected_on_mask_0[:, 0]].reshape(-1, class_scores[0].shape[2])
        point_scores_1 = class_scores[1][points_projected_on_mask_1[:, 1], points_projected_on_mask_1[:, 0]].reshape(-1, class_scores[1].shape[2])
        point_scores_2 = class_scores[2][points_projected_on_mask_2[:, 1], points_projected_on_mask_2[:, 0]].reshape(-1, class_scores[2].shape[2])
        point_scores_3 = class_scores[3][points_projected_on_mask_3[:, 1], points_projected_on_mask_3[:, 0]].reshape(-1, class_scores[3].shape[2])
        point_scores_4 = class_scores[4][points_projected_on_mask_4[:, 1], points_projected_on_mask_4[:, 0]].reshape(-1, class_scores[4].shape[2])
        #augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw[:,:5], np.zeros((lidar_raw.shape[0], class_scores[1].shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_0, -class_scores[0].shape[2]:] += point_scores_0
        augmented_lidar[true_where_point_on_img_1, -class_scores[1].shape[2]:] += point_scores_1
        augmented_lidar[true_where_point_on_img_2, -class_scores[2].shape[2]:] += point_scores_2
        augmented_lidar[true_where_point_on_img_3, -class_scores[3].shape[2]:] += point_scores_3
        augmented_lidar[true_where_point_on_img_4, -class_scores[4].shape[2]:] += point_scores_4
        augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_1, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_0_2, -class_scores[0].shape[2]:]
        augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_1_3, -class_scores[1].shape[2]:]
        augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_2_4, -class_scores[2].shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]#remove
        #augmented_lidar = self.create_cyclist(augmented_lidar)
        return augmented_lidar

    def run(self):
        image_files = os.listdir(self.root_split_path + 'image_0')
        for img_file in tqdm(image_files):
            
            sample_idx = os.path.splitext(img_file)[0]
            # points: N * 4(x, y, z, r)
            points = self.get_lidar(sample_idx)
            
            # get segmentation score from network
            scores_from_cam = []
            for i in range(5):
                scores_from_cam.append(self.get_score(sample_idx, 'image_' + str(i) + '/'))
            # scores_from_cam: H * W * 4/5, each pixel have 4/5 scores(0: background, 1: bicycle, 2: car, 3: person, 4: rider)

            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx)
            
            # paint the point clouds
            # points: N * 8
            points = self.augment_lidar_class_scores_both(scores_from_cam, points, calib_fromfile)
            
            np.save(self.save_path + ("%s.npy" % sample_idx), points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--training_path', help='your data root for the training data', required=True)
    parser.add_argument('--model_path', help='path to segmentation model', required=True)
    args = parser.parse_args()
    painter = Painter(args)
    painter.run()