import cv2
import numpy as np

class Tester:
    def find_marker(self, img, color):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        cont = self.getContours(mask)
        marker_x, marker_y = self.marker_height(cont)
        if marker_y == 0:
            marker_err = 1000
        else:
            marker_err = int((marker_y-self.height*0.5)/self.height*100)
        marker_color = self.get_color(marker_err, 20)
        return marker_x, marker_y, marker_err, marker_color

def main():
    vid = cv2.VideoCapture(0)

    while(True):
        ret, img = vid.read()
        green = [33, 38, 41, 90, 255, 255]



        cv2.imshow('webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()