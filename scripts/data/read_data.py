import cv2
import os
import pickle
import numpy as np
# from pose2D import interpolate_keypoints, prune_keypoints
from tqdm import tqdm
from skeleton_prep import DrawskeletonFrame


import matplotlib.pyplot as plt

import os
import cv2



def prune_keypoints(coords, thresh, base_thresh=0.2):
    T, N, _ = coords.shape
    mask = thresh < base_thresh
    coords[mask] = 0
    return coords




import numpy as np

# def interpolate_keypoints(coords):
#     num_frames, num_keypoints, _ = coords.shape
#     for i in range(num_frames):
#         for j in range(num_keypoints):
#             if np.all(coords[i, j] == [0, 0]):
#                 # print('The js is', j)
#                 # print('The coords ', coords[i, j])
                
#                 prev = coords[i - 1, j] if i > 0 else [0, 0]
#                 after = coords[i + 1, j] if i < num_frames - 1 else [0, 0]
#                 coords[i, j] = (prev + after) / 2

#                 # print('The coords updated', coords[i, j])
#                 # print('===========================================')
#     return coords

import numpy as np

def interpolate_keypoints(coords, window_size=3):
    num_frames, num_keypoints, _ = coords.shape
    for i in range(num_frames):
        for j in range(num_keypoints):
            if np.all(coords[i, j] == [0, 0]):
                window = [coords[k, j] for k in range(max(0, i - window_size // 2), min(num_frames, i + window_size // 2 + 1)) if np.all(coords[k, j] != [0, 0])]
                coords[i, j] = np.mean(window, axis=0) if window else [0, 0]
    return coords


def draw_pose_frames(data, frame_no = 10):


    # Extract the keypoints for the first person in the image
    pose_keypoints = data[frame_no]['people'][0]['pose_keypoints_2d']
    # print(pose_keypoints)


    # Extract x and y coordinates of the keypoints
    pose_x_coords = [pose_keypoints[i] for i in range(0, len(pose_keypoints), 3)]
    pose_y_coords = [pose_keypoints[i+1] for i in range(0, len(pose_keypoints), 3)]
    pose_thresh = [pose_keypoints[i+2] for i in range(0, len(pose_keypoints), 3)]


    #extract the face keypoints
    face_keypoints = data[frame_no]['people'][0]['face_keypoints_2d']

    # Extract x and y coordinates of the keypoints
    face_x_coords = [face_keypoints[i] for i in range(0, len(face_keypoints), 3)]
    face_y_coords = [face_keypoints[i+1] for i in range(0, len(face_keypoints), 3)]
    face_thresh = [face_keypoints[i+2] for i in range(0, len(face_keypoints), 3)]


    #extract the left hand keypoints
    left_hand_keypoints = data[frame_no]['people'][0]['hand_left_keypoints_2d']

    # Extract x and y coordinates of the keypoints
    left_hand_x_coords = [left_hand_keypoints[i] for i in range(0, len(left_hand_keypoints), 3)]
    left_hand_y_coords = [left_hand_keypoints[i+1] for i in range(0, len(left_hand_keypoints), 3)]
    left_thresh = [left_hand_keypoints[i+2] for i in range(0, len(left_hand_keypoints), 3)]

    #extract the right hand keypoints
    right_hand_keypoints = data[frame_no]['people'][0]['hand_right_keypoints_2d']

    # Extract x and y coordinates of the keypoints
    right_hand_x_coords = [right_hand_keypoints[i] for i in range(0, len(right_hand_keypoints), 3)]
    right_hand_y_coords = [right_hand_keypoints[i+1] for i in range(0, len(right_hand_keypoints), 3)]
    right_thresh = [right_hand_keypoints[i+2] for i in range(0, len(right_hand_keypoints), 3)]


    #stack values as x,y
    pose_cords = np.stack((pose_x_coords, pose_y_coords), axis=1)
    face_cords = np.stack((face_x_coords, face_y_coords), axis=1)
    left_hand_cords = np.stack((left_hand_x_coords, left_hand_y_coords), axis=1)
    right_hand_cords = np.stack((right_hand_x_coords, right_hand_y_coords), axis=1)

    #check if any value has threshold less than 0.4 in left and right hand then set the flag good to False
    good = True

    if min(left_thresh) < 0.2:
        good = False
    
    if min(right_thresh) < 0.2:
        good = False

    keypoints_output = {'pose_cords_xy': pose_cords, 
                        'face_cords_xy': face_cords, 
                        'left_hand_cords_xy': left_hand_cords, 
                        'right_hand_cords_xy': right_hand_cords,
                        'left_hand_thresh': left_thresh,
                        'right_hand_thresh': right_thresh,
                        'face_thresh': face_thresh,
                        'pose_thresh': pose_thresh,
                        'status': good}

    return keypoints_output




def  PreparePoses(pkl_path):

    # read the pkl file 
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    pose_cords = []
    face_cords = []
    left_hand_cords = []
    right_hand_cords = []

    face_thresh = []
    pose_thresh = []
    left_thresh = []
    right_thresh = []


    for i in tqdm(range(0, len(data))):
        output = draw_pose_frames(data, i)
        pose_cords.append(output['pose_cords_xy'])
        face_cords.append(output['face_cords_xy'])
        left_hand_cords.append(output['left_hand_cords_xy'])
        right_hand_cords.append(output['right_hand_cords_xy'])
        face_thresh.append(output['face_thresh'])
        pose_thresh.append(output['pose_thresh'])
        left_thresh.append(output['left_hand_thresh'])
        right_thresh.append(output['right_hand_thresh'])

    base_thresh = 0.2


    pose_cords = np.array(pose_cords)
    pose_thresh = np.array(pose_thresh)
    pruned_coords = prune_keypoints(pose_cords, pose_thresh, base_thresh)
    interpolated_coords = interpolate_keypoints(pruned_coords)

    face_cords = np.array(face_cords)
    face_thresh = np.array(face_thresh)
    pruned_face_coords = prune_keypoints(face_cords, face_thresh, base_thresh)
    interpolated_face_coords = interpolate_keypoints(pruned_face_coords)

    left_hand_cords = np.array(left_hand_cords)
    left_thresh = np.array(left_thresh)
    pruned_left_hand_coords = prune_keypoints(left_hand_cords, left_thresh, base_thresh)
    interpolated_left_hand_coords = interpolate_keypoints(pruned_left_hand_coords)

    right_hand_cords = np.array(right_hand_cords)
    right_thresh = np.array(right_thresh)
    pruned_right_hand_coords = prune_keypoints(right_hand_cords, right_thresh, base_thresh)
    interpolated_right_hand_coords = interpolate_keypoints(pruned_right_hand_coords)


    #check if there are any values of 0,0 in the interpolated coords
    # if np.any(interpolated_coords == 0):
    #     print('found 0,0 in interpolated coords')
    
    # if np.any(interpolated_face_coords == 0):
    #     print('found 0,0 in interpolated face coords')
    
    # if np.any(interpolated_left_hand_coords == 0):
    #     print('found 0,0 in interpolated left hand coords')
    
    # if np.any(interpolated_right_hand_coords == 0):
    #     print('found 0,0 in interpolated right hand coords')

    #make a dict
    output = {'pose_cords': interpolated_coords,
                'face_cords': interpolated_face_coords,
                'left_hand_cords': interpolated_left_hand_coords,
                'right_hand_cords': interpolated_right_hand_coords
                }

    return output

    


def cords_to_heatmap(cords, img_size = (1080,1720), sigma=6, categories=range(10)):
    result = np.zeros(img_size + (len(categories),), dtype='float32')
    cords = cords.tolist()
    for i in categories:
            point = cords[i]
            if point[0] == 0 or point[1] == 0:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] += np.exp(-((yy - point[1]) ** 2 + (xx - point[0]) ** 2) / (2 * sigma ** 2))
    return result




def save_heatmaps(heatmaps, path, prefix='heatmap'):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[..., i]
        filename = os.path.join(path, f"{prefix}_{i}.png")
        plt.imsave(filename, heatmap)








def GenerateHeatMaps(pkl_path, savedir):
    output = PreparePoses(pkl_path)


    for i in tqdm(range(0, len(output['pose_cords']))):
        img = DrawskeletonFrame(output['pose_cords'][i], output['face_cords'][i], output['left_hand_cords'][i], output['right_hand_cords'][i])
        cv2.imwrite('outputs/{}.png'.format(i), img)
        heatmap_pose = cords_to_heatmap(output['pose_cords'][i], categories=range(7))
        heatmap_face = cords_to_heatmap(output['face_cords'][i], categories=range(len(output['face_cords'][i])))
        heatmap_left_hand = cords_to_heatmap(output['left_hand_cords'][i], categories=range(len(output['left_hand_cords'][i])))
        heatmap_right_hand = cords_to_heatmap(output['right_hand_cords'][i], categories=range(len(output['right_hand_cords'][i])))

        # print(heatmap_pose.shape)
        # print(heatmap_face.shape)
        # print(heatmap_left_hand.shape)
        # print(heatmap_right_hand.shape)

        heatmap_pose  = heatmap_pose.mean(axis=-1)
        heatmap_face  = heatmap_face.mean(axis=-1)

        right_index_finger = [heatmap_right_hand[:,:,i] for i in [0,1,2,3,4]]
        right_middle_finger = [heatmap_right_hand[:,:,i] for i in [0,5,6,7,8]]
        right_ring_finger = [heatmap_right_hand[:,:,i] for i in [0,9,10,11,12]]
        right_pinky_finger = [heatmap_right_hand[:,:,i] for i in [0,13,14,15,16]]
        right_thumb = [heatmap_right_hand[:,:,i] for i in [0,17,18,19,20]]

        left_index_finger = [heatmap_left_hand[:,:,i] for i in [0,1,2,3,4]]
        left_middle_finger = [heatmap_left_hand[:,:,i] for i in [0,5,6,7,8]]
        left_ring_finger = [heatmap_left_hand[:,:,i] for i in [0,9,10,11,12]]
        left_pinky_finger = [heatmap_left_hand[:,:,i] for i in [0,13,14,15,16]]
        left_thumb = [heatmap_left_hand[:,:,i] for i in [0,17,18,19,20]]


        right_index_finger = np.stack(right_index_finger, axis=-1)
        right_middle_finger = np.stack(right_middle_finger, axis=-1)
        right_ring_finger = np.stack(right_ring_finger, axis=-1)
        right_pinky_finger = np.stack(right_pinky_finger, axis=-1)
        right_thumb = np.stack(right_thumb, axis=-1)

        left_index_finger = np.stack(left_index_finger, axis=-1)
        left_middle_finger = np.stack(left_middle_finger, axis=-1)
        left_ring_finger = np.stack(left_ring_finger, axis=-1)
        left_pinky_finger = np.stack(left_pinky_finger, axis=-1)
        left_thumb = np.stack(left_thumb, axis=-1)


        #take a mean of all the heatmaps
        right_index_finger  = right_index_finger.mean(axis=-1)
        right_middle_finger  = right_middle_finger.mean(axis=-1)
        right_ring_finger  = right_ring_finger.mean(axis=-1)
        right_pinky_finger  = right_pinky_finger.mean(axis=-1)
        right_thumb  = right_thumb.mean(axis=-1)

        left_index_finger  = left_index_finger.mean(axis=-1)
        left_middle_finger  = left_middle_finger.mean(axis=-1)
        left_ring_finger  = left_ring_finger.mean(axis=-1)
        left_pinky_finger  = left_pinky_finger.mean(axis=-1)
        left_thumb  = left_thumb.mean(axis=-1)



        all_heatmaps = [heatmap_pose, 
                        heatmap_face, 
                        right_index_finger, 
                        right_middle_finger, 
                        right_ring_finger, 
                        right_pinky_finger, 
                        right_thumb, 
                        
                        left_index_finger, 
                        left_middle_finger, 
                        left_ring_finger, 
                        left_pinky_finger,
                        left_thumb,
                        img[:,:,0],
                        img[:,:,1],
                        img[:,:,2]]


        # for heatmap in all_heatmaps:
        #     print('---------------')
        #     print(heatmap.shape)

        
        all_heatmaps = np.stack(all_heatmaps, axis=-1)
        # print(all_heatmaps.shape)
        #save the info as pickle

        #make the dir if it doesn't exist
        if not os.path.exists(os.path.join(savedir, 'heatmaps')):
            os.makedirs(os.path.join(savedir, 'heatmaps'))

        with open(os.path.join(savedir, 'heatmaps', f'{i}.pkl'), 'wb') as f:
            pickle.dump(all_heatmaps, f)

        # break



if __name__ == '__main__':
    pkl_path = '/home/saiteja/Desktop/untitled folder/SignLangaugeProject/outputs/dataset_preparation/raw_vids_dataset/1176340/1176340A-0/1176340A-0.pkl'
    save_path = os.path.dirname(pkl_path)
    GenerateHeatMaps(pkl_path,save_path)