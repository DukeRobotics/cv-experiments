import torch
import math

# format for tensor
# tensor([[a,b,c,d],[a,b,c,d]]) -> 2 boxes
# a, b, c, d = xmin, ymin, xmax, ymax

# assuming that t_left and t_right are tensors with only one box each
def depth_estimate(t_left, t_right, width):
    ## check dimensions of t_left and t_right
    if t_left.size()[0] == 0 or t_right.size()[0] == 0:
        print('there is no box! returning 0')
        return 0
    if t_left.size()[0] != 1 or t_right.size()[0] != 1:
        print('there are too many boxes! returning 0')
        return 0
    ## parameters
    threshold = 4 # threshold in percent to determine that the bounding box is "touching" an edge
    threspix = width*threshold/100.0
    B = 68.58 # in mm, distance between two cameras
    angle_of_view = 40 # angle of view in degrees, estimated
    angle = math.pi*angle_of_view/180 # converted to radians
    f = width/2/math.tan(angle/2) # focal length in terms of pixels

    ## calculating the depth
    # x-coordinates for the left camera's box, x1 is left edge, x2 is right edge
    leftx1 = float(t_left[0][0])
    leftx2 = float(t_left[0][2])
    # x-coordinates for the right camera's box, x1 is left edge, x2 is right edge
    rightx1 = float(t_right[0][0])
    rightx2 = float(t_right[0][2])
    # if either left or right end of the bounding box "touches" an edge, just take the other one
    if leftx1 < threspix or rightx1 < threspix:
        leftcenter = leftx2
        rightcenter = rightx2
    elif leftx2 > width-threspix or rightx2 > width-threspix:
        leftcenter = leftx1
        rightcenter = rightx1
    else: # in case the box doesn't touch any edge, take the center of the boxes
        leftcenter = (leftx1+leftx2)/2
        rightcenter = (rightx1+rightx2)/2
    # disparity (in pixels) needed to estimate depth
    disparity = leftcenter - rightcenter
    # calculate the depth with given parameter
    my_constant = 2.15 # theoretically this should be 1.0 if the cameras are perfectly aligned
    depth = f*B/disparity * my_constant # multiply by certain constant
    if depth <= 0 :
        print('negative depth calculated, retunring 0')
        return 0
    return depth # return depth in mm

def main():
    # testing code
    tl = torch.tensor([[755, 485, 860, 590]])
    tr = torch.tensor([[685, 485, 790, 590]])
    print(tl.size()[0])
    print(depth_estimate(tl, tr, 1210))

if __name__ == '__main__':
    main()
    pass
