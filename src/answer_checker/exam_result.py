import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import *
import re
import pandas as pd
from typing import Any, List, Tuple, Dict

def preprocess_image(image: np.ndarray, iter: int) -> np.ndarray:
    """
    Pre-process the image for binary processing. 

    Args: 
        image (np.ndarray): The image in pixel format. 

    Returns: 
        image_erode (np.ndarray): The image in inverted binary format.  

    """

    # Rotate image
    h, w = image.shape[:2]
    
    # TODO: This might not be needed
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) if h < w else image
        
    # Preprocess image color
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_threshold = cv2.threshold(image_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image_invert = cv2.bitwise_not(image_threshold)
    image_bin = cv2.erode(image_invert, None, iterations=iter)

    return image_bin

def locate_rect_contour(image_bin: np.ndarray) -> List[np.ndarray]:
    """
    Locate the relevant contours by finding contours with 4 sides. 

    Args: 
        image_bin (np.ndarray): The image in binary format. 

    Returns: 
        cont_rectangle_sorted (List): The list of contours with 4 sides.            

    """

    # Find contours of answer tables
    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours for answer table
    cont_rectangle = []
    for contour in contours: 
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
        if len(approx) == 4:        # if contour has 4 sides (rectangle)
            cont_rectangle.append(approx)
            
    # Sort the contours by their x coordinates
    cont_rectangle_sorted = sorted(cont_rectangle, key=lambda x: [cv2.boundingRect(x)][0])

    return cont_rectangle_sorted

def locate_answer_tables(cont_rectangle_sorted: List[np.ndarray]) -> List[np.ndarray]:
    """
    Locate the answer tables by finding the contours with maximum area and removing redundant contours. 

    Args: 
        cont_rectangle (List): The list of contours with 4 sides.            
    
    Returns: 
        answer_tables (List): The list of contours of answer tables. 

    """
    
    # Find the max area of the contours
    max_area = 0
    cont_area = {}
    for contour in cont_rectangle_sorted: 
        area = int(cv2.contourArea(contour))
        cont_area[area] = contour
        if area > max_area: 
            max_area = area

    # Sort out rectangles with area much smaller than max area (5% buffer)
    cont_sorted = []
    for area, contour in cont_area.items():
        if area > max_area * 0.95:
            cont_sorted.append(contour)

    # Keep only the rectangles with x-coordinates far apart (too close: redundant)
    answer_tables = [cont_sorted[0]]                # keep the first rectangle

    i = 0
    j = i + 1
    while(j < len(cont_sorted)):                    # avoid index overflow
        difference = abs(cv2.boundingRect(cont_sorted[i])[0] - cv2.boundingRect(cont_sorted[j])[0])
        if difference > 10:
            i = j                                   # when found a new answer table, skip to compare next
            answer_tables.append(cont_sorted[i])    # add the newly found answer table
        j += 1                                      # keep comparing

    return answer_tables

def order_points(pts): 
    """
    Find the coordinates of the 4 corners of the contours. 

    Args: 
        pts (np.ndarray): A np.ndarray which represents the side of contour. 

    Returns: 
        rect (List): A list which represents the coordinates of the 4 corners. 
        The output format should be: [top-left, top-right, bottom-right, bottom-left]
    """
    
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def distance_points(p1, p2): 
    """
    Find the distance between coordinates. 

    Args: 
        p1, p2 (Int): A pair of coordinates. 

    Returns: 
        distance (Int): The distance in pixel. 
    """
    distance = int(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)

    return distance

def perspective_transform(image, answer_tables) -> List[np.ndarray]:
    """
    Perform perspective transformation for the region of interest ("roi"), i.e. answer tables, identified. 

    Args: 
        image (np.adarray): The original image. 
        answer_tables (List[np.ndarray]): The list of roi contours identified.

    Returns: 
        answer_tables_perspective (List[np.ndarray]): The list of cropped images of roi. 
    """
    w = image.shape[1]
    answer_tables_perspective = []
    for table in answer_tables: 
        cont_table_pts = order_points(table)
    
        # Find the width and height of the answer table
        w_coor = distance_points(cont_table_pts[0], cont_table_pts[1])     # distance of top-left and top-right
        h_coor = distance_points(cont_table_pts[0], cont_table_pts[3])     # distance of top-left and bottom-left

        ratio_new = h_coor / w_coor

        w_new = int(w * 0.9)
        h_new = int(w * ratio_new)

        # Apply perspective transform
        pts1 = np.float32(cont_table_pts)
        pts2 = np.float32([[0, 0], [w_new, 0], [w_new, h_new], [0, h_new]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image_perspective_correct = cv2.warpPerspective(image, matrix, (w_new, h_new))
        answer_tables_perspective.append(image_perspective_correct)
        
    return answer_tables_perspective

def extract_answer_box(answer_tables_perspective: List[np.ndarray]) -> List[List[float]]:
    """
    1. Divide the answer tables into 60 answer boxes. 
    2. Translate the answer boxes to the proportion of black pixels within the box. 

    Args: 
        answer_tables_perspective (List[np.ndarray]): The list of cropped images of roi. 

    Returns: 
        answers (List[List[int]]): 
            The list of answer boxes by their proportion of black pixels within the box. 
    """
    answers = []
    for i_table in range(len(answer_tables_perspective)): 
        # Pre-process image color
        table_height, table_width = answer_tables_perspective[i_table].shape[:2]
        table_bw = cv2.cvtColor(answer_tables_perspective[i_table], cv2.COLOR_BGR2GRAY)
        table_threshold = cv2.threshold(table_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        table_erode = cv2.erode(table_threshold, None, iterations=5)

        # Divide the answer tables by each question
        box_height = 30
        box_width = 5
        box_outline_h = int((table_height // box_height) * 0.15)
        box_outline_w = int((table_width // box_width) * 0.15)

        for ih in range(box_height): 
            answer_row = []

            for iw in range(box_width): 
                x = int(table_width / box_width * iw)
                y = int(table_height / box_height * ih)
                h = int(table_height / box_height)
                w = int(table_width / box_width)
                
                # Find the fill proportion of each box
                box = table_erode[y + box_outline_h: y + h - box_outline_h, x + box_outline_w: x + w - box_outline_w]
                n_black_px = np.sum(box == 0)
                answer_row.append(n_black_px / (h * w))

            answers.append(answer_row)              # keep the proportion of black pixels of each box

    return answers

def compose_result(answers: List[List[float]]) -> List[Tuple[int]]:
    """
    Determine the answer choices based on the proportion of black pixels within the box. 
    

    Args: 
        answers (List[List[float]]): 
            The list of answer boxes by their proportion of black pixels within the box. 

    Returns: 
        results (List[Tuple[int]]): 
            The list of answers choices of each question. 
            The choices are represented by: blank - 0; choice - 1. 
            Example of 1 question: (1, 0, 0, 0, 0)

    """
    # Define the thresholds for identifying filled boxes
    quantile_filled = np.quantile(np.array(answers), 0.75)          # filled box: top 20% of population with 5% buffer
    quantile_template = np.quantile(np.array(answers), 0.875)       # template: median of filled box distribution
    # ... outliers

    # Locate the filled boxes and template
    results = []
    for answer in answers:
        result_row = [0, 0, 0, 0, 0]
        max_box = 0
        # max_a = 0
        min_diff = 1
        # min_a = 0
        for i, box in enumerate(answer):
            # Record the box with max proportion for fallback
            if box > max_box: 
                max_box = box
                max_a = i

            # Locate filled boxes (min difference from quantile_template)
            if box >= quantile_filled:      
                diff = abs(box - quantile_template)
                if diff < min_diff: 
                    min_diff = diff
                    min_a = i
        
        result_row[min_a] = 1

        # ... tag outliers as 2 (?)

        # Fallback for questions where no answers was within threshold: add the box with max proportion of fill
        if sum(result_row) == 0: 
            result_row[max_a] = 1
        
        result_row = tuple(result_row)
        results.append(result_row) 

    return results

# Main function: directory
def get_result(image: np.ndarray) -> List[Tuple[int]]: 
    """
    The main function: translate image to answer choices. 

    Args: 
        image (np.adarray): The original image. 

    Returns: 
        results (List[Tuple[int]]): 
            The list of answers choices of each question. 
            The choices are represented by: blank - 0; choice - 1. 
    """
    check = 0                   # check the number of answer tables detected
    iter = [3, 0]               # 3: erosion, 0: no erosion 
    for i in range(2): 
        image_bin = preprocess_image(image, iter[i])
        cont_rectangle = locate_rect_contour(image_bin)
        answer_tables = locate_answer_tables(cont_rectangle)

        # Error detection: only proceed if there are 2 tables
        check = len(answer_tables)      
        if check == 2: 
            answer_tables_perspective = perspective_transform(image, answer_tables)
            answers = extract_answer_box(answer_tables_perspective)
            results = compose_result(answers)
            
            return results      # break the loop if there are 2 answer tables
            
        elif i < 1:
            continue            # continue to try another method
        
        else: 
            return None         # if both methods could not detect 2 tables

def get_model_answer(doc: str) -> List[str]: 
    """
    Retrieve the model answer from .txt file. 

    Args: 
        doc (str): The file path of the file containing model answer.

    Returns: 
        model_answer_alph (List[str]): The list of answer choices in alphabets. 
    """
    # Extract the answer in doc
    doc_answer = open(doc, "r")
    model_answer_alph = doc_answer.read()
    model_answer_alph = re.sub(r"\s", "", model_answer_alph)    # remove formats
    model_answer_alph = model_answer_alph.rsplit(".")
    model_answer_alph = [x[0] for x in model_answer_alph][1:]   # extract alphabets    
    doc_answer.close()

    return model_answer_alph

def format_model_answer(model_answer_alph: List[str]) -> List[Tuple[int]]: 
    """
    Translate the model answer in alphabets into blank - 0; choice - 1. 

    Args: 
        model_answer_alph (List[str]): The list of answer choices in alphabets. 

    Returns: 
        model_answer (List[Tuple[int]]): The list of answer choices in specific foramt.
        The format is represented as: blank - 0; choice - 1.  
    """
    # Format model answers
    map_alph = {'A': (1, 0, 0, 0, 0), 
        'B': (0, 1, 0, 0, 0), 
        'C': (0, 0, 1, 0, 0), 
        'D': (0, 0, 0, 1, 0), 
        'E': (0, 0, 0, 0, 1)}
    
    model_answer = []
    for alph in model_answer_alph: 
        _a = map_alph[alph]
        model_answer.append(_a)

    return model_answer

def create_output(model_answer: List[Tuple[int]], all_result: Dict) -> pd.DataFrame:
    """
    Create output with answer choices and score of each student. 

    Args: 
        model_answer (List[Tuple[int]]): The list of answer choices in specific foramt.
        all_result (Dict): The dictionary of answer choices of all students. 

    Returns: 
        df (pd.DataFrame): A pd dataframe which lists out: 
        1. answer choices (in alphabets) and score of each student, and
        2. model answer (in alphabets). 
    """
    output = all_result.copy()
    output['Model Answer'] = model_answer

    df = pd.DataFrame(output).T
    df.columns = [f'Q{i + 1}' for i in df.columns]
    df.index.name = 'SID'

    # Calculate prelimenary score
    for index, row in df.iterrows(): 
        score = 0
        for i_ans in range(len(model_answer)): 
            _score = all([a==b for a, b in zip(model_answer[i_ans], output[index][i_ans])])
            score += 1 if _score else 0

        df.loc[index, 'Score'] = score

    # Transform answer to alphabet for display
    map_vec = {(1, 0, 0, 0, 0): 'A', 
            (0, 1, 0, 0, 0): 'B', 
            (0, 0, 1, 0, 0): 'C', 
            (0, 0, 0, 1, 0): 'D', 
            (0, 0, 0, 0, 1): 'E'}

    df = df.map(lambda x: map_vec.get(x, x))            # if not match, stay the same

    df = df.sort_index(key=lambda x: x.str.isdigit())   # sort: model answer > sid (ascending)

    return df