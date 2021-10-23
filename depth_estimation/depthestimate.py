import cv2
import math

# size of squares for large checkerboard: 108 mm
# Hyungbin Jun

def main():
    # load images
    leftimg = cv2.imread('left.jpg')
    rightimg = cv2.imread('right.jpg')
    # getting parameters
    threshold = 10 # threshold to determine the bounding box is "touching" an edge
    B = 68.58 # in mm, distance between 2 cameras
    height, width, channels = leftimg.shape
    angle_of_view = 40 # in degrees, estimated
    angle = math.pi*angle_of_view/180 # converted to radians
    f = width/2/math.tan(angle/2) # calculating focal length in terms of pixels

    # setting bounding box points manually [(left p1), (left p2), (right p1), (right p2)]
    names = ['checker', 'stuff', 'board']
    objects = []
    checker = [(755, 485), (860, 590), (685, 485), (790, 590)]
    objects.append(checker)
    stuff = [(500, 400), (590, 450), (435, 390), (525, 440)]
    objects.append(stuff)
    board = [(745, 170), (1205, 630), (675, 180), (1180, 640)]
    objects.append(board)

    # draw bounding boxes
    for i in range(len(objects)):
        # points for the left camera, p1 is topleft, p2 is bottomright
        leftp1 = objects[i][0]
        leftp2 = objects[i][1]
        # points for the right camera, p1 is topleft, p2 is bottomright
        rightp1 = objects[i][2]
        rightp2 = objects[i][3]
        # if either left or right end of the bounding box "touches" an edge, just take the other one
        if leftp1[0] < threshold or rightp1[0] < threshold:
            leftcenter = leftp2[0]
            rightcenter = rightp2[0]
        elif leftp2[0] > (width - threshold) or rightp2[0] > (width - threshold):
            leftcenter = leftp1[0]
            rightcenter = rightp1[0]
        else: # in case the box doesn't touch any edge, take the center of the boxes
            leftcenter = (leftp1[0]+leftp2[0])/2
            rightcenter = (rightp1[0]+rightp2[0])/2
        # disparity (in pixels) needed to estimate depth
        disparity = leftcenter - rightcenter
        # calculate the depth with given parameter
        depth = f*B/disparity * 2.15 # multipy by certain constant
        print("estimated depth for " + names[i]+ " (mm): ", depth)

        cv2.rectangle(leftimg, leftp1, leftp2, (0, 255, 0), 3)
        cv2.rectangle(rightimg, rightp1, rightp2, (0, 255, 0), 3)
    # show images
    cv2.imshow('left', leftimg)
    cv2.imshow('right', rightimg)
    cv2.setWindowProperty('left', cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty('right', cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    print('distance: ', a)


if __name__ == '__main__':
    # Test an image processing algorithm
    main()
    pass
