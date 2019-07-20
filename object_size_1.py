from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import pickle
import settings,os
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())
with open(settings.CALIB_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    mtx = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
# load the image, convert it to grayscale, and blur it slightly
for image_file in os.listdir(args["image"]):
        if image_file.endswith("jpg"):
            image = cv2.imread(os.path.join(args["image"], image_file))
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 502, 376)
            print(image_file)
            # for i in range(2):
            #     image=cv2.pyrDown(image)
            image=cv2.undistort(image, mtx, dist_coeffs)
            # cv2.imshow("ss",image)
            # cv2.waitKey(0)
            # cv2.imwrite("img1.png",image)
            # cv2.destroyAllWindows()
            # print(image.shape)
            # image=cv2.pyrDown(image)
            # print(image.shape)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:,:,0] #this one shows good results best results till now.
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #this one shows good results best results till now.

            # cv2.imshow("image",gray)
            # cv2.waitKey(0)
            gray = cv2.GaussianBlur(gray, (9, 9), 0)
            # cv2.imshow("image",gray)
            # cv2.waitKey(0)
            # cv2.imwrite("img1.png",gray)
            # cv2.destroyAllWindows()
            ret,gray=cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
            gray = cv2.bitwise_not(gray)
            # cv2.imshow("image",gray)
            # cv2.waitKey(0)
            # cv2.imwrite("img1.png",gray)
            # cv2.destroyAllWindows()
            # perform edge detection, then perform a dilation + erosion to
            # close gaps in between object edges
            edged = cv2.Canny(gray, 50, 100)
            # cv2.imshow("image",edged)
            # cv2.waitKey(0)
            edged = cv2.dilate(edged, None, iterations=5)
            # cv2.imshow("image",edged)
            # cv2.waitKey(0)
            edged = cv2.erode(edged, None, iterations=5)
            # cv2.imshow("image",edged)
            # cv2.waitKey(0)
            # cv2.imwrite("img2.png",edged)
            # cv2.destroyAllWindows()
            # find contours in the edge map
            # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)

            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
            pixelsPerMetric = None
            i=3
            for c in cnts:
                if i<=4:
                    i=i+1
                    continue
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 1117:
                    continue

                # compute the rotated bounding box of the contour
                orig = image.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                # cv2.imshow("image",orig)
                # cv2.waitKey(0)
                # # loop over the original points and draw them
                # for (x, y) in box:
                #     cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                # cv2.imshow("image",orig)
                # cv2.waitKey(0)
                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                # cv2.imshow("image",orig)
                # cv2.waitKey(0)
                #
                # # draw lines between the midpoints
                # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
                # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
                # cv2.imshow("image",orig)
                # cv2.waitKey(0)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # print(dA)
                # print(dB)
                # print(2*dA + 2*dB)
                # print("----------")
                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric
                # (in this case, inches)
                if pixelsPerMetric is None:
                    pixelsPerMetric = dB / args["width"]

                # compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric

                # draw the object sizes on the image
                # cv2.putText(orig, "{:.1f}in".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
                # cv2.putText(orig, "{:.1f}in".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

                # show the output image
                draw=image.copy()
                cv2.drawContours(draw,c,-1,(255,255,0),3)
                perimeter = cv2.arcLength(c, True)
                perimeter = perimeter / pixelsPerMetric
                cv2.putText(draw, "{:.3f}".format(perimeter),(int(tltrX-50), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 0, 255), 2)
                cv2.imshow("image",draw)
                # cv2.imwrite("img"+str(i)+".png",draw)
                cv2.waitKey(0)
                i=i+1
                # cv2.imshow("Image", orig)
                # cv2.waitKey(0)
