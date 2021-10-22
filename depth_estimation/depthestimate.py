import cv2
import math

# size of squares for large checkerboard: 108 mm
# Hyungbin Jun

def main():
    # load images
    leftimg = cv2.imread('left.jpg')
    rightimg = cv2.imread('right.jpg')
    # getting parameters
    height, width, channels = leftimg.shape
    print(height,width)
    angle_of_view = 20 # in degrees, estimated
    angle = math.pi*angle_of_view/180 # converted to radians
    f = width/2/math.tan(angle/2)
    # draw bounding box manually
    objects = []
    checker = [(755, 485), (860, 590), (685, 485), (790, 590)]
    objects.append(checker)
    stuff = [(500, 400), (590, 450), (435, 390), (525, 440)]
    objects.append(stuff)
    for i in range(len(objects)):
        leftp1 = objects[i][0]
        leftp2 = objects[i][1]
        rightp1 = objects[i][2]
        rightp2 = objects[i][3]
        # setting up parameters
        B = 68.58 # mm, estimated
        leftcenter = (leftp1[0]+leftp2[0])/2
        rightcenter = (rightp1[0]+rightp2[0])/2
        disparity = leftcenter - rightcenter
        # calculate the depth with given parameter
        depth = f*B/disparity
        print("estimated depth in mm: ", depth)

        cv2.rectangle(leftimg, leftp1, leftp2, (0,255,0), 3)
        cv2.rectangle(rightimg, rightp1, rightp2, (0,255,0), 3)
    # show images
    cv2.imshow('left', leftimg)
    cv2.imshow('right', rightimg)
    cv2.setWindowProperty('left', cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty('right', cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)

if __name__ == '__main__':
    # Test an image processing algorithm
    main()
    pass
