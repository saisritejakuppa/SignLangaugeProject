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




import numpy as np
import cv2

def DrawskeletonFrame(pose_cords, face_cords, left_hand_cords , right_hand_cords):

    posepairs, facepoints, eyes_brows, fingers = PairsOfKeypoints() 

    # lines plots
    img = np.zeros((1080, 1720,  3), np.uint8)

    for no,posePP in enumerate(posepairs):
        part1 = posePP[0]
        part2 = posePP[1]

        part1 = (pose_cords[part1][0], pose_cords[part1][1])
        part2 = (pose_cords[part2][0], pose_cords[part2][1])

        #convert to int
        part1 = (int(part1[0]), int(part1[1]))
        part2 = (int(part2[0]), int(part2[1]))

        #dont draw if any values is 0,0
        if part1 == (0,0) or part2 == (0,0):
            continue

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

                #dont draw if any values is 0,0
        if part1 == (0,0) or part2 == (0,0):
            continue

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

                #dont draw if any values is 0,0
        if part1 == (0,0) or part2 == (0,0):
            continue

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

                # dont draw if any values is 0,0
        if part1 == (0,0) or part2 == (0,0):
            continue



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

        #         #dont draw if any values is 0,0
        if part1 == (0,0) or part2 == (0,0):
            continue

        #draw lines on the image with various colours each time
        cv2.line(img, part1, part2, (0, 0, 255), 2)


    #crop the image to 512*512 so that the skeleton parts lies inside the image using the bounding box
    # get a bounding box for the image
    return img
