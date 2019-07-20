import cv2
import numpy as np
import math
import pickle
import settings,os
from BoundaryDetection import getSegMask

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

def checkCircular(con,area,perimeter):

    circularity = 4*math.pi*(area/(perimeter*perimeter))
    return circularity


def checkRectangle(con,area):

    rect = cv2.minAreaRect(con)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area_rect = box[1][0]*box[1][1]
    
    return area/area_rect,box



def detectShapes(thresh,img_color):

    orig = thresh.copy()
    orig_color=img_color.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(thresh,contours,-1,(255,255,0),3)
    contours_area = []
    # contours_area2=[]

    # calculate area and filter into new array
    flag=1
    for con in contours:
        area = cv2.contourArea(con)
        #Adjust the area as required for th holes.
        if 50< area < 10000000:
            #alternate values are taken as the countours in an edged image has two countours fo one hole
            #be sure to change this part if you make changes to the way countour are extracted.
            if flag==0:
                contours_area.append(con)#this on gives hte inner diameter.
                flag=1
            else:
                # contours_area.append(con)#this one gives the outer diameter
                # contours_area2.append(con)
                flag=0


    contours_allowed = []
    # imgorig=img_color.copy()

# check if contour is of circular shape
    for con in contours_area:
        temp_img = orig
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        M = cv2.moments(con)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if perimeter == 0:
            continue
        circularity = checkCircular(con,area,perimeter)
      

        if 0.7 < circularity < 1.2:
            contours_allowed.append(con)
            cv2.drawContours(img_color,[con],0,(255,255,0),1)
            perimeter=perimeter/3.7735849056603774
            perimeter=perimeter/math.pi
            print("{:.3f}".format(perimeter))
            cv2.putText(img_color, "{:.3f}".format(perimeter), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ss",img_color)
            cv2.waitKey(0)
            #Create the mask required for the circular hole
            mask = np.zeros_like(orig) # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, [con],0, 255,-1) # Draw filled contour in mask
            out = np.zeros_like(orig_color) # Extract out the object and place into output image
            out[mask == 255] = orig_color[mask == 255]#Keep the colour same as the original one where mask is there
            #Now crop the region of intrest by finding a bounding rectangle.
            (y,x)=np.where(mask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out = out[topy:bottomy+1, topx:bottomx+1]
            # cv2.imshow("ss",out)
            # cv2.waitKey(0)
            #Some morphological operations carried out to get more accurate masks
            #TODO:-Get the accurate mask back to the original image an get the perimeter from it.
            #Be sure that this process just works for some cases to get accurate results for others the result from the previous mask should be used.
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            cv2.imshow("ss",out)
            cv2.waitKey(0)
            ret,out2=cv2.threshold(out,20,255,cv2.THRESH_BINARY)
            cv2.imshow("ss",out2)
            cv2.waitKey(0)
            out2=out2-out
            cv2.imshow("ss",out2)
            cv2.waitKey(0)
            ret,out2=cv2.threshold(out2,180,255,cv2.THRESH_BINARY)
            cv2.imshow("ss",out2)
            cv2.waitKey(0)
            continue
    # count = 0
    # while os.path.isfile("./output holes/imgF"+str(count)+".png"):  # increase number until file not exists
    #     count += 1
    # cv2.imwrite("./output holes/imgF"+str(count)+".png",img_color)
#     contours_allowed2 = []
#     print("--------------------------------------------------------------------------------------------------")
# # check if contour is of circular shape
#     for con in contours_area2:
#         temp_img = orig
#         perimeter = cv2.arcLength(con, True)
#         area = cv2.contourArea(con)
#         M = cv2.moments(con)
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#
#         if perimeter == 0:
#             continue
#         circularity = checkCircular(con,area,perimeter)
#
#
#         if 0.7 < circularity < 1.2:
#             contours_allowed2.append(con)
#             cv2.drawContours(imgorig,[con],0,(0,0,255),5)
#             perimeter=perimeter/3.7735849056603774
#             perimeter=perimeter/math.pi
#             print("{:.3f}".format(perimeter))
#             cv2.putText(imgorig,"{:.3f}".format(perimeter), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#             cv2.imshow("ss",imgorig)
#             cv2.waitKey(0)
#             continue
        # approx = cv2.approxPolyDP(con,0.01*perimeter,True)
        # print (len(approx))
        # if len(approx) == 4:
        #     contours_allowed.append(con)

    # print "Final countours"
    # cv2.drawContours(img_color,contours_allowed,-1,(255,255,0),1)
    # cv2.imshow("ss",img_color)
    # cv2.waitKey(0)
    # cv2.drawContours(imgorig,contours_allowed2,-1,(0,0,255),1)
    # cv2.imshow("ss",imgorig)
    # cv2.waitKey(0)

with open(settings.CALIB_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    mtx = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (3008,4016), 1, (3008,4016))
cv2.namedWindow('ss',cv2.WINDOW_NORMAL)
cv2.resizeWindow('ss', 502, 376)
img = cv2.imread("/home/jbmai/Downloads/Hole Sample Photos 19.07.19-20190719T053840Z-001/Hole Sample Photos 19.07.19/NG Hole Size Photos/IMG_20190718_175845.jpg")
# img = image_resize(img,height=1080)

img = cv2.undistort(img, mtx, dist_coeffs, None, newcameramtx)
# crop the image
x, y, w, h = roi
img = img[y:y+h, x:x+w]
img_orig = img.copy()
cv2.imshow("ss",img)
cv2.waitKey(0)
# kernel = np.ones((5,5),np.float32)/25

# img = cv2.GaussianBlur(img,(3,3),0)
# img = cv2.GaussianBlur(img,(15,15),0)
# img = cv2.medianBlur(img,3)

img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img = img[:,:,0]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.bitwise_not(img)
# kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
# filtered = cv2.filter2D(img, -1, kernel)
# cv2.imshow("ss",filtered)
# cv2.waitKey(0)
# # filtered=filtered - np.min(filtered)
# # filtered = filtered * (255.0/np.max(filtered))
#
# cv2.imshow("ss",filtered)
# cv2.waitKey(0)
# sharpened = img + filtered
# cv2.imshow("ss",sharpened)
# cv2.waitKey(0)
# sharpened = sharpened - np.min(sharpened)
# sharpened = sharpened * (255.0/np.max(sharpened ))
# # img = cv2.GaussianBlur(img,(9,9),0)
# cv2.imshow("ss",sharpened)
# cv2.waitKey(0)
# ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

# ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
# img = cv2.bitwise_not(img)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# cv2.imshow("ss",img)
# cv2.waitKey(0)
# kernel = np.ones((3,3),np.float32)

# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.Canny(img,100,200)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# img = cv2.dilate(img, kernel, iterations=5)
#
# img = cv2.erode(img, kernel, iterations=2)
# kernel = np.ones((2,2),np.float32)
# img = cv2.erode(img, kernel, iterations=6)
# cv2.imshow("ss",img)
#
# cv2.waitKey(0)

# kernel = kernel = np.ones((3,3),np.float32)/9
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# mask = getSegMask(img_orig)
# print(img.shape)
# img = cv2.bitwise_and(img, img, mask = mask)
detectShapes(img,img_orig)


# cv2.imshow("sss",img)
# cv2.waitKey(0)



