import cv2
import numpy as np
import logging
import math
import RPi.GPIO as GPIO          
from time import sleep

def stabilize_steering_angle(
          curr_steering_angle,
          new_steering_angle,
          num_of_lane_lines,
          max_angle_deviation_two_lines=5,
          max_angle_deviation_one_lane=1):
    if num_of_lane_lines == 2 :
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        max_angle_deviation = max_angle_deviation_one_lane
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
            + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle
def compute_steering_angle(frame, lane_lines):
    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

  
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset) 
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    logging.debug('new steering angle: %s' % steering_angle)
    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
          
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def detect_line_segments(cropped_edges):
    rho = 1  
    angle = np.pi / 180 
    min_threshold = 10 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)


    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_edges(frame):   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    edges = cv2.Canny(mask, 200, 400)

    return edges

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def detect_lane(frame):
    
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  
    y2 = int(y1 * 1 / 2)  


    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary) 
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))



    return lane_lines

def laneDetection(image):
    
    cv2.imshow('image',image)
    
    #turns all of the same shade of color the same color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)
    
    #finds all blue and changes it to white while everything else is black
    bottom_blue = np.array([60,40,40])
    top_blue = np.array([150,255,255])
    mask = cv2.inRange(hsv, bottom_blue, top_blue)
    cv2.imshow('mask', mask)
    
    #gets the edges of the mask
    edges = cv2.Canny(mask, 200,400)
    cv2.imshow('edges', edges)
    
    #cropped edges
    #cutoff half of the vision just incase there is another blue object
    height, width = edges.shape
    blackSpace = np.zeros_like(edges)
    #square to mask the to
    square = np.array([[(0,height*.5),(width, height*.5),(width, height),(0,height)]],np.int32)
    #combine square with blackspace
    cv2.fillPoly(blackSpace,square,255)
    #matches all the ones in edges with ones in blackSpace
    edgesCutOff = cv2.bitwise_and(edges, blackSpace)
    cv2.imshow('edgesCutOff', edgesCutOff)
    
    #houghlinesp
    line_segments = cv2.HoughLinesP(edgesCutOff,1, np.pi/180,10,np.array([]), minLineLength = 4, maxLineGap=8)
    #print(line_segments)

def main():
    cam = cv2.VideoCapture(0)
    in1 = 24
    in2 = 23
    en = 25
    
    right = False
    left = False
    current = 90
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1,GPIO.OUT)
    GPIO.setup(in2,GPIO.OUT)
    GPIO.setup(en,GPIO.OUT)
    
    x = input()
    if(x=='r'):
        while(cam.isOpened()):
            if(left):
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.HIGH)
            elif(right):
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.LOW)
            else:
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.HIGH)
            ret,image = cam.read()
            cv2.imshow('original',image)
            laneDetection(image)
            try:
                lane_lines = detect_lane(image)
            except TypeError:
                print("oiapjo")
                GPIO.cleanup()
                break
            lane_lines_image = display_lines(image, lane_lines)
            cv2.imshow("lane lines", lane_lines_image)
            
            steering_angle = compute_steering_angle(image,lane_lines)
            stable = stabilize_steering_angle(steering_angle,current, len(lane_lines))
            print(stable)
            if(stable < 85):
                GPIO.output(in1,GPIO.LOW)
                GPIO.output(in2,GPIO.HIGH)
                left = False
                right = True
            elif(stable>95):
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.LOW)
                left = True
                right = False
            else:
                GPIO.output(in1,GPIO.HIGH)
                GPIO.output(in2,GPIO.HIGH)
                left = False
                right = False
            middle = display_heading_line(image,stable)
            cv2.imshow("middle",middle)
            current = steering_angle
            if cv2.waitKey(1) == ord('q'):
                break
        GPIO.cleanup()
        cam.release()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
