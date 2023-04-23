# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import math
# video =cv2.VideoCapture(r'C:\Users\jakub\Pictures\IMG_0138 - Trim - Trim_linia.mp4')
# video =cv2.VideoCapture(r'C:\Users\jakub\Pictures\IMG_0138 - Trim.mp4')
video =cv2.VideoCapture(r'C:\Users\jakub\Pictures\IMG_0138 - Trimpaleta.mp4')
# video =cv2.VideoCapture(r'C:\Users\jakub\Pictures\IMG_0138 - Trimkat.mp4')
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def areaFilter(minArea, inputImage):

    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def distance_btw_points(a, b):
    x_diff = a[0] - b[0]
    y_diff = a[1] - b[1]

    return math.sqrt(x_diff ** 2 + y_diff ** 2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    while True:
        # wczytaj klatkę z wideo
        ret, frame = video.read()

        # sprawdź, czy klatka została prawidłowo wczytana
        if not ret:
            break

        img = frame.copy()
        lowerValues = np.array([110, 50, 50])
        upperValues = np.array([130, 255, 255])



        # Convert the image to HSV:
        hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create the HSV mask
        mask = cv2.inRange(hsvImage, lowerValues, upperValues)

        # minArea = 50
        # mask = areaFilter(minArea, mask)




        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # img_with_contours = img.copy()
        # for cnt in contours:
        #
        #  cv2.drawContours(img_with_contours, [cnt], -1, (0, 255, 0), 3)




        # gray = mask.copy()
        # ret, thresh = cv2.threshold(gray, 50, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, 1, 2)
        #
        # for cnt in contours:
        #     x1, y1 = cnt[0][0]
        #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #     if len(approx) >=3 and len(approx) <12:
        #     # if len(approx) == 4 :
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         ratio = float(w) / h
        #         if ratio >= 0.9 and ratio <= 1.1:
        #             img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
        #             cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #         else:
        #             cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #             img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
        #


        for cnt in contours:
            x1, y1 = cnt[0][0]


            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])


            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if cv2.contourArea(cnt) > 5000.0:
                cv2.circle(img, leftmost, 5, (0, 0, 255), -1)
                cv2.circle(img, rightmost, 5, (0, 255, 0), -1)
                cv2.circle(img, topmost, 5, (255, 0, 0), -1)
                cv2.circle(img, bottommost, 5, (255, 255, 0), -1)
                if(distance_btw_points(leftmost,rightmost) > 225 and distance_btw_points(topmost,bottommost) > 225):

                    # print(ratio)
                    print(len(approx))
                    if len(approx) < 15:
                        # if (ratio >= 1.1 and ratio <= 2.5) :# or (ratio >= 0.6 and ratio <= 0.95):
                        img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
                        cv2.putText(img, str(distance_btw_points(topmost,bottommost)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Wideo', img)

        # przerwij odtwarzanie, gdy użytkownik naciśnie klawisz 'q'
        if cv2.waitKey(1) == ord('q'):
            break
            cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()



