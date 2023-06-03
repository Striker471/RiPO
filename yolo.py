import cv2
import numpy as np
import math
import torch

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

video = cv2.VideoCapture(r'scena7.mp4')
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path=r'/Users/julia/Downloads/best (1).pt')


# def distance_btw_points(a, b):
#     x_diff = a[0] - b[0]
#     y_diff = a[1] - b[1]
#
#     return math.sqrt(x_diff ** 2 + y_diff ** 2)
#
# if __name__ == '__main__':
#     while True:
#         ret, frame = video.read()
#
#         if not ret:
#             break
#
#         results = model(frame)
#
#         # Render the image with bounding boxes, labels and so on.
#         rendered_img = results.render()
#
#         # Use the first image in the list.
#         rendered_img = rendered_img[0]
#
#         # Convert color space from BGR to RGB
#         # rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
#
#         cv2.imshow('Wideo', rendered_img)
#
#         if cv2.waitKey(1) == ord('q'):
#             break
#
# video.release()
# cv2.destroyAllWindows()


def distance_btw_points(a, b):
    x_diff = a[0] - b[0]
    y_diff = a[1] - b[1]

    return math.sqrt(x_diff ** 2 + y_diff ** 2)


if __name__ == '__main__':
    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_copy = frame.copy()
        model_yolov5.conf = 0.50  # confidence threshold (0-1)
        model_yolov5.iou = 0.45  # NMS IoU threshold (0-1)

        # Detekcja palet za pomocą modelu YOLOv5
        results = model_yolov5(frame_copy)
        results.conf = 0.5

        # Renderowanie obrazu z bounding boxami i etykietami
        rendered_img = results.render()
        rendered_img = rendered_img[0]

        # Detekcja pól odkładczych
        # Konwertowanie do HSV
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerValues = np.array([110, 50, 50])
        upperValues = np.array([130, 255, 255])
        mask = cv2.inRange(hsvImage, lowerValues, upperValues)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x1, y1 = cnt[0][0]

            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if cv2.contourArea(cnt) > 5000.0:
                cv2.circle(rendered_img, leftmost, 5, (0, 0, 255), -1)
                cv2.circle(rendered_img, rightmost, 5, (0, 255, 0), -1)
                cv2.circle(rendered_img, topmost, 5, (255, 0, 0), -1)
                cv2.circle(rendered_img, bottommost, 5, (255, 255, 0), -1)
                if (distance_btw_points(leftmost, rightmost) > 225 and distance_btw_points(topmost, bottommost) > 225):
                    if len(approx) < 15:
                        rendered_img = cv2.drawContours(rendered_img, [cnt], -1, (0, 255, 255), 3)
                        cv2.putText(rendered_img, 'Storage area', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 0),
                                    2)

        cv2.imshow('Wideo', rendered_img)

        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
