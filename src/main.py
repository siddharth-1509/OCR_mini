import image_preprocessing as ip
import image_to_text as t2t
import open_image_file
from shutil import copyfile
import imutils as im
import cv2 as c


def main():
    print("Path of an image for pre-processing",end=': ')
    filename=open_image_file.return_file_name()
    print(filename)
    file_out=ip.process(filename,filename.split('/')[-1])
    print("Image preprocessing Completed")
    print("Path of an image for print the text in the image",end=': ')
    print(file_out)
    t2t.image_2_text(file_out)
main()