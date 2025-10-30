import cv2
import numpy as np
from math import *

# Rotate the image degree counterclockwise (original size)
def rotateImage(src, degree):
    # The center of rotation is the center of the image
    h, w = src.shape[:2]
    # Calculate the two-dimensional rotating affine transformation matrix
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # print("Rotate Matrix: ")
    # print(RotateMatrix)

    # Affine transformation, the background color is filled with GREEN so that the rotation can be easily understood
    rotate1 = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(0, 255, 0))
    #plot_fig(rotate1)
    # Affine transformation, the background color is filled with white
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))

    # Padding
    bg_color = [255, 255, 255]
    pad_image_rotate = cv2.copyMakeBorder(rotate,100,100,100,100,cv2.BORDER_CONSTANT,value=bg_color)

    return pad_image_rotate

class ImgCorrect():
    def __init__(self, img):
        self.img = img
        self.h, self.w, self.channel = self.img.shape
        # print("Original images h & w -> | w: ",self.w, "| h: ",self.h)
        if self.w <= self.h:
            self.scale = 700 / self.w
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.scale = 700 / self.h
            self.img = cv2.resize(self.img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        print("Resized Image by Padding and Scaling:")
        #plot_fig(self.img)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_lines(self):
        print("Gray Image:")
        #plot_fig(self.gray)
        ret, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("bin",binary)
        print("Inverse Binary:")
        #plot_fig(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular structure
        # print("Kernel for dialation:")
        # print(kernel)
        binary = cv2.dilate(binary, kernel)  # dilate
        print("Dilated Binary:")
        #plot_fig(binary)
        edges = cv2.Canny(binary, 50, 200)
        print("Canny edged detection:")
        #plot_fig(edges)

        # print("Edge 1: ")
        # cv2.imshow("edges", edges)

        self.lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        # print(self.lines)
        if self.lines is None:
            print("Line segment not found")
            return None

        lines1 = self.lines[:, 0, :]  # Extract as 2D
        # print(lines1)
        imglines = self.img.copy()
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(imglines, (x1, y1), (x2, y2), (0, 255, 0), 3)
        print("Probabilistic Hough Lines:")
        #plot_fig(imglines)
        return imglines

    def search_lines(self):
      lines = self.lines[:, 0, :]  # extract as 2D
    
      number_inexist_k = 0
      sum_pos_k45 = number_pos_k45 = 0
      sum_pos_k90 = number_pos_k90 = 0
      sum_neg_k45 = number_neg_k45 = 0
      sum_neg_k90 = number_neg_k90 = 0
      sum_zero_k = number_zero_k = 0

      for x in lines:
          if x[2] == x[0]:
              number_inexist_k += 1
              continue
          #print(degrees(atan((x[3] - x[1]) / (x[2] - x[0]))), "pos:", x[0], x[1], x[2], x[3], "Slope:",(x[3] - x[1]) / (x[2] - x[0]))
          degree = degrees(atan((x[3] - x[1]) / (x[2] - x[0])))
          # print("Degree or Slope of detected lines : ",degree)
          if 0 < degree < 45:
              number_pos_k45 += 1
              sum_pos_k45 += degree
          if 45 <= degree < 90:
              number_pos_k90 += 1
              sum_pos_k90 += degree
          if -45 < degree < 0:
              number_neg_k45 += 1
              sum_neg_k45 += degree
          if -90 < degree <= -45:
              number_neg_k90 += 1
              sum_neg_k90 += degree
          if x[3] == x[1]:
              number_zero_k += 1

      max_number = max(number_inexist_k, number_pos_k45, number_pos_k90, number_neg_k45,number_neg_k90, number_zero_k)
      # print("Num of lines in different Degree range ->")
      # print("Not a Line: ",number_inexist_k, "| 0 to 45: ",number_pos_k45, "| 45 to 90: ",number_pos_k90, "| -45 to 0: ",number_neg_k45, "| -90 to -45: ",number_neg_k90, "| Line where y1 equals y2 :",number_zero_k)
    
      if max_number == number_inexist_k:
          return 90
      if max_number == number_pos_k45:
          return sum_pos_k45 / number_pos_k45
      if max_number == number_pos_k90:
          return sum_pos_k90 / number_pos_k90
      if max_number == number_neg_k45:
          return sum_neg_k45 / number_neg_k45
      if max_number == number_neg_k90:
          return sum_neg_k90 / number_neg_k90
      if max_number == number_zero_k:
          return 0

    def rotate_image(self, degree):
        """
        Positive angle counterclockwise rotation
        :param degree:
        :return:
        """
        # print("degree:", degree)
        if -45 <= degree <= 0:
            degree = degree  # #negative angle clockwise
        if -90 <= degree < -45:
            degree = 90 + degree  # positive angle counterclockwise
        if 0 < degree <= 45:
            degree = degree  # positive angle counterclockwise
        if 45 < degree <= 90:
            degree = degree - 90  # negative angle clockwise
        print("DSkew angle: ", degree)

        # degree = degree - 90
        height, width = self.img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(
            cos(radians(degree))))  # This formula refers to the previous content
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        # print("Height :",height)
        # print("Width :",width)
        # print("HeightNew :",heightNew)
        # print("WidthNew :",widthNew)

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # rotate degree counterclockwise
        # print("Mat Rotation (Before): ",matRotation)
        matRotation[0, 2] += (widthNew - width) / 2
        # Because after rotation, the origin of the coordinate system is the upper left corner of the new image, so it needs to be converted according to the original image
        matRotation[1, 2] += (heightNew - height) / 2
        # print("Mat Rotation (After): ",matRotation)

        # Affine transformation, the background color is filled with white
        imgRotation = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        # Padding
        pad_image_rotate = cv2.warpAffine(self.img, matRotation, (widthNew, heightNew), borderValue=(0, 255, 0))
        #plot_fig(pad_image_rotate)

        return imgRotation
    
def dskew(line_path, img):
    img_loc = line_path + img
    im = cv2.imread(img_loc)

    # Padding
    bg_color = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value=bg_color)

    imgcorrect = ImgCorrect(pad_img)
    lines_img = imgcorrect.img_lines()
    # print(type(lines_img))
    
    if lines_img is None:
        rotate = imgcorrect.rotate_image(0)
    else:
        degree = imgcorrect.search_lines()
        rotate = imgcorrect.rotate_image(degree)


    return rotate

# Degree conversion
def DegreeTrans(theta):
    res = theta / np.pi * 180
    # print(res)
    return res

# Calculate angle by Hough transform
def CalcDegree(srcImage,canny_img):
    lineimage = srcImage.copy()
    lineimg = srcImage.copy()
    # Detect straight lines by Hough transform
    # The fourth parameter is the threshold, the greater the threshold, the higher the detection accuracy
    try:
        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200)
        # print("HoughLines: ")
        # cv2_imshow(lines)
        # Due to different images, the threshold is not easy to set, because the threshold is set too high, so that the line cannot be detected, the threshold is too low, the line is too much, the speed is very slow
        theta_sum = 0
        rho_sum = 0
        sum_x1 = sum_x2 = sum_y1 = sum_y2 = 0
        # Draw each line segment in turn
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                # print("theta:", theta, " rho:", rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # print("a: ",a, " b: ",b, " x0: ",x0, " y0: ",y0, " x1: ",x1, " y1: ",y1, " x2: ",x2, " y2: ",y2)
                # Only select the smallest angle as the rotation angle
                sum_x1+=x1
                sum_x2+=x2
                sum_y1+=y1
                sum_y2+=y2
                rho_sum += rho
                theta_sum += theta
                cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

        
        print("HoughLines: ")
        #plot_fig(lineimage)
        print()

        pt1 = (sum_x1//len(lines), sum_y1//len(lines))
        pt2 = (sum_x2//len(lines), sum_y2//len(lines))
        
        # print("Sum of thetas: ",theta_sum)
        # print("lines: ",lines)
        average = theta_sum / len(lines)
        # print("Avg. Theta: ",average)
        angle = DegreeTrans(average) - 90
        # print("Avg. Angle: ",angle)
        print("Skewed Angle: ",angle)
        average_rho = rho_sum / len(lines)
        # print("Avg. rho: ",average_rho)

        # print(pt1,pt2)
        print('Draw best fit line with full:')
        # h, w = lineimg.shape[:2]
        # pt2 = (w,h)
        # print("Cordinates of the best fit line: ",pt1,pt2)
        cv2.line(lineimg, pt1, pt2, (0,0,255), 2)
        #plot_fig(lineimg)
        # cv2_imshow(lineimg)

        return angle
    except:
        angle = 0.0
        return angle

def ready_for_rotate(line_path, img, directory):
    print()
    print("Image :: ",img)
    rotate_line = f"./{directory}/Rotated_line_by_HaughLine_Affine/"
    # os.mkdir(rotate_line)

    rotate_line_Dskew = f"./{directory}/DSkew/"
    # os.mkdir(rotate_line_Dskew)

    rotate_line_Haughline = f"./{directory}/HaughLine_Affine/"
    img_loc = line_path + img
    image = cv2.imread(img_loc)

    org_width = image.shape[1]
    org_height = image.shape[0]

    img1 = image
    im_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print("Gray Image: ")
    #plot_fig(im_gray)

    edges = cv2.Canny(im_gray,50,150,apertureSize=3)
    print("Canny Image: ")
    #plot_fig(edges)

    degree = CalcDegree(image,edges)
    
    if degree == 0.0:
        rotate = dskew(line_path, img)
        print("Rotated Image by DSkew: ")
        #plot_fig(rotate)
        print()
        
        filename1 = rotate_line_Dskew + img
        cv2.imwrite(filename1, rotate)
        filename = rotate_line + img
        cv2.imwrite(filename, rotate)
    else:
        rotate = rotateImage(image, degree)
        print("Rotated Image by Houghline Affine transform: ")
        #plot_fig(rotate)
        print()

        filename2 = rotate_line_Haughline + img
        cv2.imwrite(filename2, rotate)
        filename = rotate_line + img
        cv2.imwrite(filename, rotate)