import cv2 as c
import numpy as np
import imutils as im
from skimage.filters import threshold_local
from PIL import Image

def transform(image, axis):
	r = np.zeros((4, 2), dtype = "float32")
	s = axis.sum(axis = 1)
	r[0] = axis[np.argmin(s)]
	r[2] = axis[np.argmax(s)]
	diff = np.diff(axis, axis = 1)
	r[1] = axis[np.argmin(diff)]
	r[3] = axis[np.argmax(diff)]
	(top_left, top_right, bottom_right, bottom_left) = r
	wi_A = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
	wi_B = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
	width_max = max(int(wi_A), int(wi_B))
	he_A = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
	he_B = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
	height_max = max(int(he_A), int(he_B))
	dst = np.array([
		[0, 0],
		[width_max - 1, 0],
		[width_max - 1, height_max - 1],
		[0, height_max - 1]], dtype = "float32")
	M = c.getPerspectiveTransform(r, dst)
	gray_image = c.warpPerspective(image, M, (width_max, height_max))
	return gray_image


def process(fname,save_output):
    image=c.imread(fname)
    image=c.medianBlur(image, 3)
    sec_image=image.copy()
    ratio=image.shape[0]/500.0
    image=im.resize(image,height=500)
    b_w=c.cvtColor(image,c.COLOR_BGR2GRAY)
    b_w=c.GaussianBlur(b_w,(5,5),0)
    edge_d=c.Canny(b_w, 75, 200)
    ed=c.findContours(edge_d.copy(), c.RETR_LIST, c.CHAIN_APPROX_SIMPLE)
    ed=im.grab_contours(ed)
    ed=sorted(ed,key=c.contourArea,reverse=True)[:5]
    try:
        for x in ed:
            p=c.arcLength(x,True)
            app=c.approxPolyDP(x,0.02*p,True)
            if len(app)==4:
                s=app
                break
        c.drawContours(image,[s],-1,(255,0,0),4)
        image=im.resize(image,height=650)
        c.imshow("EDGE_Detect",image)
        c.waitKey(0)
        image=transform(sec_image,s.reshape(4, 2) * ratio)

    except UnboundLocalError:
        print("No boundary detected!")
        path_to_save='D:\\Coding\\Mine_OCR\\image_after_preprocessing\\'+save_output
        c.imwrite(path_to_save,image)
        return path_to_save
    gray_image=c.cvtColor(image,c.COLOR_BGR2GRAY)
    thres=threshold_local(gray_image,11,offset=10,method="gaussian")
    gray_image=(gray_image>thres).astype("uint8")*255
    c.imshow("Output",im.resize(gray_image,height=650))
    c.waitKey(0)
    path_to_save='D:\\Coding\\Mine_OCR\\image_after_preprocessing\\'+save_output
    c.imwrite(path_to_save,gray_image)
    return path_to_save