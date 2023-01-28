
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import cv2


def DrawHeatMapsForPoints(points, width, height, scale = 5, sigma = 5):

    x_points = points[0]
    y_points = points[1]

    # Create an image with the same size as the original image
    heatmap = np.zeros((height, width))

    # Iterate through the keypoints and increment the value of the corresponding pixels in the heatmap
    for i in range(0, len(x_points), 1):
        if x_points[i]==0 or y_points[i]== 0:
            continue        
        heatmap[int(y_points[i]), int(x_points[i])] += scale

    # Normalize the heatmap
    heatmap = heatmap / np.max(heatmap)

    # Apply a Gaussian filter to the heatmap
    heatmap = gaussian_filter(heatmap, sigma)

    return heatmap



def GetHeatMapInfo(pinky, left_hand_cords):
    #left Pink
    x_pink = []
    y_pink = []
    for no,PinkPP in enumerate(pinky):
        part1, part2 = PinkPP[0], PinkPP[1]
        part1, part2 = left_hand_cords[part1], left_hand_cords[part2]
        x_pp = [part1[0], part2[0]]
        y_pp = [part1[1], part2[1]]

        x_pink.append(part1[0])
        y_pink.append(part1[1])

        if no == len(pinky)-1:
            x_pink.append(part2[0])
            y_pink.append(part2[1])

    return x_pink, y_pink




class keypoints:

    def __init__(self, openpose_json_path):
        self.openpose_json_path = openpose_json_path

    
    def ReadJson(self):
        # Load the JSON file
        with open(self.openpose_json_path , 'r') as f:
            data = json.load(f)
        return data

    
    def KeyPointsExtract(self, person_number, data, frameid):

        # Extract the keypoints for the first person in the image
        pose_keypoints = data[person_number]['frames'][str(frameid)]['people'][0]['pose_keypoints_2d']


        self.height = data[person_number]['height']
        self.width = data[person_number]['width']

        # Extract x and y coordinates of the keypoints
        pose_x_coords = [pose_keypoints[i] for i in range(0, len(pose_keypoints), 3)]
        pose_y_coords = [pose_keypoints[i+1] for i in range(0, len(pose_keypoints), 3)]
        pose_thresh = [pose_keypoints[i+2] for i in range(0, len(pose_keypoints), 3)]


        #extract the face keypoints
        face_keypoints = data[person_number]['frames'][str(frameid)]['people'][0]['face_keypoints_2d']

        # Extract x and y coordinates of the keypoints
        face_x_coords = [face_keypoints[i] for i in range(0, len(face_keypoints), 3)]
        face_y_coords = [face_keypoints[i+1] for i in range(0, len(face_keypoints), 3)]
        face_thresh = [face_keypoints[i+2] for i in range(0, len(face_keypoints), 3)]


        #extract the left hand keypoints
        left_hand_keypoints = data[person_number]['frames'][str(frameid)]['people'][0]['hand_left_keypoints_2d']

        # Extract x and y coordinates of the keypoints
        left_hand_x_coords = [left_hand_keypoints[i] for i in range(0, len(left_hand_keypoints), 3)]
        left_hand_y_coords = [left_hand_keypoints[i+1] for i in range(0, len(left_hand_keypoints), 3)]
        left_thresh = [left_hand_keypoints[i+2] for i in range(0, len(left_hand_keypoints), 3)]

        #extract the right hand keypoints
        right_hand_keypoints = data[person_number]['frames'][str(frameid)]['people'][0]['hand_right_keypoints_2d']

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


        return [pose_cords, face_cords, left_hand_cords, right_hand_cords], good


    
    def DrawHeatMaps(self, points):
        pose_cords, face_cords, left_hand_cords, right_hand_cords = points

        posepairs, facepoints, eyes_brows, fingers = PairsOfKeypoints() 
        thumb, index, mid, ring, pinky = fingers[:4], fingers[4:8], fingers[8:12], fingers[12:16], fingers[16:20]

        # print('The pose pairs', len(posepairs))
        # print(posepairs)

        heatmaps = []


        #pose 
        for no,PosePP in enumerate(posepairs):
            part1, part2 = PosePP[0], PosePP[1]
            part1, part2 = pose_cords[part1], pose_cords[part2]
            x_pp = [part1[0], part2[0]]
            y_pp = [part1[1], part2[1]]
            heatmap= DrawHeatMapsForPoints([x_pp, y_pp], self.width, self.height)
            heatmaps.append(heatmap)

            #save the heatmaps
            # plt.imsave(f'data/{no}_pose_heatmap.png', heatmap)

        
        #heatmap for faces
        x_faces = []
        y_faces = []
        for no,FacePP in enumerate(facepoints):
            part1, part2 = FacePP[0], FacePP[1]
            part1, part2 = face_cords[part1], face_cords[part2]
            x_pp = [part1[0], part2[0]]
            y_pp = [part1[1], part2[1]]
            x_faces.append(part1[0])
            y_faces.append(part1[1])

            if no == len(facepoints)-1:
                x_faces.append(part2[0])
                y_faces.append(part2[1])
        heatmap= DrawHeatMapsForPoints([x_faces, y_faces], self.width, self.height)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/{no}_face_heatmap.png', heatmap)


        #eyes brows+ eyes
        x_eyes = []
        y_eyes = []
        for no,EyesPP in enumerate(eyes_brows):
            part1, part2 = EyesPP[0], EyesPP[1]
            part1, part2 = face_cords[part1], face_cords[part2]
            x_pp = [part1[0], part2[0]]
            y_pp = [part1[1], part2[1]]

            x_eyes.append(part1[0])
            y_eyes.append(part1[1])

            if no == len(eyes_brows)-1:
                x_eyes.append(part2[0])
                y_eyes.append(part2[1])
        
        heatmap = DrawHeatMapsForPoints([x_eyes, y_eyes], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/eyes_heatmap.png', heatmap)


        x_pink, y_pink = GetHeatMapInfo(pinky, left_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_pink, y_pink], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/lpink_heatmap.png', heatmap)

        x_pink, y_pink = GetHeatMapInfo(pinky, right_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_pink, y_pink], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/rpink_heatmap.png', heatmap)

        x_ring, y_ring = GetHeatMapInfo(ring, left_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_ring, y_ring], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/lring_heatmap.png', heatmap)

        x_ring, y_ring = GetHeatMapInfo(ring, right_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_ring, y_ring], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/rring_heatmap.png', heatmap)

        x_mid, y_mid = GetHeatMapInfo(mid, left_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_mid, y_mid], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)
        
        #save the heatmaps
        # plt.imsave(f'data/lmid_heatmap.png', heatmap)

        x_mid, y_mid = GetHeatMapInfo(mid, right_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_mid, y_mid], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/rmid_heatmap.png', heatmap)

        x_index, y_index = GetHeatMapInfo(index, left_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_index, y_index], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/lindex_heatmap.png', heatmap)

        x_index, y_index = GetHeatMapInfo(index, right_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_index, y_index], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/rindex_heatmap.png', heatmap)

        x_thumb, y_thumb = GetHeatMapInfo(thumb, left_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_thumb, y_thumb], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/lthumb_heatmap.png', heatmap)

        x_thumb, y_thumb = GetHeatMapInfo(thumb, right_hand_cords)
        heatmap = DrawHeatMapsForPoints([x_thumb, y_thumb], self.width, self.height, sigma = 3, scale = 5)
        heatmaps.append(heatmap)

        #save the heatmaps
        # plt.imsave(f'data/rthumb_heatmap.png', heatmap)


        pose_cords, face_cords, left_hand_cords, right_hand_cords = points

        posepairs, facepoints, eyes_brows, fingers = PairsOfKeypoints() 

        # lines plots
        img = np.zeros((self.height, self.width, 3), np.uint8)

        for no,posePP in enumerate(posepairs):
            part1 = posePP[0]
            part2 = posePP[1]

            part1 = (pose_cords[part1][0], pose_cords[part1][1])
            part2 = (pose_cords[part2][0], pose_cords[part2][1])

            #convert to int
            part1 = (int(part1[0]), int(part1[1]))
            part2 = (int(part2[0]), int(part2[1]))

            #draw lines on the image with various colours each time
            cv2.line(img, part1, part2, (255, 0, 0), 2)
        
        for no,facePP in enumerate(facepoints):
            part1 = facePP[0]
            part2 = facePP[1]

            part1 = (face_cords[part1][0], face_cords[part1][1])
            part2 = (face_cords[part2][0], face_cords[part2][1])

            #convert to int
            part1 = (int(part1[0]), int(part1[1]))
            part2 = (int(part2[0]), int(part2[1]))

            #draw lines on the image with various colours each time
            cv2.line(img, part1, part2, (0, 255, 0), 2)
        
        for no,eyePP in enumerate(eyes_brows):
            part1 = eyePP[0]
            part2 = eyePP[1]

            part1 = (face_cords[part1][0], face_cords[part1][1])
            part2 = (face_cords[part2][0], face_cords[part2][1])

            #convert to int
            part1 = (int(part1[0]), int(part1[1]))
            part2 = (int(part2[0]), int(part2[1]))

            #draw lines on the image with various colours each time
            cv2.line(img, part1, part2, (0, 255, 0), 2)

        
        for no,handPP in enumerate(fingers):
            part1 = handPP[0]
            part2 = handPP[1]

            part1 = (left_hand_cords[part1][0], left_hand_cords[part1][1])
            part2 = (left_hand_cords[part2][0], left_hand_cords[part2][1])

            #convert to int
            part1 = (int(part1[0]), int(part1[1]))
            part2 = (int(part2[0]), int(part2[1]))

            #draw lines on the image with various colours each time
            cv2.line(img, part1, part2, (0, 0, 255), 2)

        for no,handPP in enumerate(fingers):
            part1 = handPP[0]
            part2 = handPP[1]

            part1 = (right_hand_cords[part1][0], right_hand_cords[part1][1])
            part2 = (right_hand_cords[part2][0], right_hand_cords[part2][1])

            #convert to int
            part1 = (int(part1[0]), int(part1[1]))
            part2 = (int(part2[0]), int(part2[1]))

            #draw lines on the image with various colours each time
            cv2.line(img, part1, part2, (0, 0, 255), 2)

        
        #save the lines
        # cv2.imwrite('pose_lines.png', img)

        heatmaps = np.array(heatmaps)
        #change the shape of the heatmaps
        # heatmaps = np.transpose(heatmaps, (1, 2, 0))

        # change the dims of the image to stack and to give inp to torch
        img = np.transpose(img, (2, 0, 1))

        #concatenate the channel 
        img = np.concatenate((img, heatmaps), axis = 0)

        #crop the image to 512*512 so that the skeleton parts lies inside the image using the bounding box
        # get a bounding box for the image
        return img


def PairsOfKeypoints():
    posepairs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7]]
    facepoints = [  [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 8],
                    [8, 9],
                    [9, 10],
                    [10, 11],
                    [11, 12],
                    [12, 13],
                    [13, 14],
                    [14, 15],
                    [15, 16],
                 ] +  \
                 [ [27, 28],
                   [28, 29],
                   [29, 30]
                 ] + \
                [[31, 32],
                [32, 33],
                [33, 34],
                [34, 35]
                ] + \
                 [[48, 49],
                 [49, 50],
                [50, 51],
                [51, 52],
                [52, 53],
                [53, 54],
                [54, 55],
                [55, 56],
                [56, 57],
                [57, 58],
                [58, 59],
                [58,48]] + \
                    [[60, 61],
                    [61, 62],
                    [62, 63],
                    [63, 64],
                    [64, 65],
                    [65, 66],
                    [66, 67],
                    [67,60]]

    eyes_brows = [[36, 37],
                [37, 38],
                [38, 39],
                [39, 40],
                [40, 41],
                [41,36]] + \
                    [[42, 43],
                [43, 44],
                [44, 45],
                [45, 46],
                [46, 47],
                [47,42]]

    
    fingers = [[0,1], [1,2], [2,3], [3,4],
            [0,5], [5,6], [6,7], [7,8],
            [0,9], [9,10], [10,11], [11,12],
            [0,13], [13,14], [14,15], [15,16],
            [0,17], [17,18], [18,19], [19,20]
           ]

    return posepairs, facepoints, eyes_brows, fingers

                                


import cv2

def SaveFrame(video_path, frame_number, output_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Set the current frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the current frame
    ret, frame = cap.read()

    # Save the frame to a file
    cv2.imwrite(output_path,  frame)

    # Release the VideoCapture object
    cap.release()

    return



import mediapipe as mp


def GetHandKeyPoints(image, threshold = 0.5):
    mp_hands = mp.solutions.hands

    # Run MediaPipe Hands.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=threshold) as hands:

        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

        print(results.multi_handedness)



# path = 'D:\projects\governement_project\SignLangaugeGAN\SignLangaugeProject\samples\heatmap_22.png'
# img = cv2.imread(path)
# GetHandKeyPoints(img)














# Path: main.py
# openpose_json_path = '/home/saiteja/extra/signgan/1583882.openpose.json'
# vidname = '/home/saiteja/extra/signgan/SignLangaugeRecognition/scripts/keypointsgeneration/1583882.mp4'

# process = keypoints(openpose_json_path)

# data = process.ReadJson()


# from tqdm import tqdm

# count = 0
# #make the dir imgs, heatmaps if not present

# imgspath = '/home/saiteja/extra/signgan/SignLangaugeRecognition/output/imgs'
# heatmapspath = '/home/saiteja/extra/signgan/SignLangaugeRecognition/output/heatmaps'

# import os
# if not os.path.exists(imgspath):
#     os.makedirs(imgspath)

# if not os.path.exists(heatmapspath):
#     os.makedirs(heatmapspath)


# for i in tqdm(range(0, 41864, 50)):
#     points, dataset_good = process.KeyPointsExtract(0, data,i)

#     if dataset_good:
#         count += 1
#         # print('The dataset is ', dataset_good)
#         try:
#             heatmaps = process.DrawHeatMaps(points)
#             SaveFrame(vidname, i, f'{imgspath}/img_frame_'+str(i)+'.png')

#             #save the heatmaps as npy file
#             np.save(f'{heatmapspath}/heatmaps_'+str(i)+'.npy', heatmaps)

#         except:
#             print('The frame number is ', i)
#             continue

# print('The count is ', count)








# save each heatmap into datafolder
# for no, heatmap in enumerate(heatmaps):

#     if no< 3:
#         cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap)
    
#     else:
#         #get the max and min value inside the heatmap
#         max_val = np.max(heatmap)
#         min_val = np.min(heatmap)

#         #normalize the heatmap  
#         heatmap = (heatmap - min_val)/(max_val - min_val)

#         #save a generated heatmap
#         cv2.imwrite('data/heatmap_'+str(no)+'.png', heatmap*255)


