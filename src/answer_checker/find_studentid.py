import cv2
import pytesseract
import numpy as np
import PIL
from scipy import stats
from pathlib import Path
from typing import Any, List, Tuple, Optional, Union
from .strhub_reader import StrhubReader


def find_mode(res: Dict[str, str]):
    r"""This applies majority vote to results.

    .. note::
        This function ignores anything with non-digit characters.
    
    Args:
        res (dict):
            Format should be {'name': 'text'}
    
    Return:
        str
    
    """
    #* remove non digits
    ignore = []
    for k, row in res.items():
        if re.match(r"[^0-9]", row) is not None:
            ignore.append(k)
            
        if len(row) != 10:
            ignore.append(k)
            
    arrs = [list(v) for k, v in res.items() if not k in ignore]
    arrs = np.array(arrs).astype(int)

    m = stats.mode(arrs)
    m = ''.join(m.mode.astype(str))
    return m

def group_locations(locations, min_radius):
    r"""Group locations of bbox obtained from template match. This sets
    a threshold to the minimum distance between two bounding boxes.
    
    Args:
        locations (np.ndarray):
            This should be an array with shape (n_samples, 2). Assumes
            the first column is x and second column is y.
            
    Returns:
        np.ndarray:
    
    .. notes:
        Note that the returned array becomes a float array.
    
    """
    x = locations[:, 0][ : , None]
    dist_x = x - x.T
    y = locations[:, 1][ : , None]
    dist_y = y - y.T
    dist = np.sqrt(dist_x**2 + dist_y**2)
    np.fill_diagonal(dist, min_radius+1)
    too_close = np.nonzero(dist <= min_radius)
    groups = []
    points = np.arange(len(locations))
    i = 0
    j = 0
    while i < len(points):
        groups.append([points[i]])
        for j in range(len(too_close[0])):
            if too_close[0][j] == points[i]:
                groups[i].append(too_close[1][j])
                points = np.delete(points, np.nonzero(points == too_close[1][j]))
        i += 1

    new_locations = []
    for group in groups:
        new_locations.append(np.mean(locations[group], axis=0))

    return np.array(new_locations)


def find_student_id_bbox(img: np.ndarray) -> List[int]:
    """Find the bounding box of the phrase 'Student ID' in an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        list: A list of bounding box coordinates in the format [x, y, w, h], 
              or an empty list if the phrase is not found.

    Raises:
        FileNotFoundError: If the image_path does not correspond to a valid file.
        
    
    Examples:
    
    >>> 
    
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to detect words and their bounding boxes
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=custom_config)
    
    # Iterate through each detected word
    for i in range(len(details['text'])):
        if "student" in details['text'][i].lower():
            # Extract the bounding box coordinates
            x, y, w, h = (details['left'][i], details['top'][i],
                           details['width'][i], details['height'][i])
            
            # Offset to exclude 'Student ID'
            x += int(w *1.6)
            
            # Expand this selection 
            w = int(w * 4.3)
            h = int(h * 3.5)
            
            # Offset for viewership
            x -= int(w * .03)
            y -= int(h * .35)
            
            return [x, y, w, h]
    
    # Return an empty list if the phrase is not found
    return details
    raise ArithmeticError(f"No bounding box detected :{details}")


def template_match(image_patch: np.ndarray, 
                   template: Union[str, np.ndarray], 
                   threshold:float =0.8) -> List[Tuple[int]]:
    r"""
    
    Args:
        image_patch (np.ndarray): 
            Input patch to perform template match.
        template (path or array):
            Template image or path to template image.
        threshold (float, Optional):
            Default to 8.
    
    Return:
        List[Tuple[int]]:
            Organized as [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
    """
    if isinstance(template, (str, Path)):
        if not Path(template).is_file():
            raise FileNotFoundError(f"Cannot found template {template}")
        # load template
        template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    # convert image to gray
    img_g = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    # perform template match
    res = cv2.matchTemplate(img_g,template,cv2.TM_CCOEFF)
    # normalize result
    res = res / res.max()
    
    loc = np.where(res >= threshold)
    
    out = []
    for pt in zip(*loc[::-1]):
        out.append((pt[0], pt[1], w, h))
    return out

def read_text(image_patch: np.ndarray, reader: StrhubReader) -> str:
    r"""Read the text from a patch
    
    Args:
        image_patch (np.ndarray):
            The patch that contain text
    
    """
    if isinstance(image_patch, np.ndarray):
        image_patch = StrhubReader.cvmat2pil(image_patch)
    
    # Read using Strhub model
    text, conf = reader.read_text(image_patch)
       
    # * Check if reading is correct
    
    # check # digit
    
    # check all digits

    # check start digit sequence 
    
    # * return if everything pass
    return text


def get_sid(img: cv2.UMat, 
            reader: StrhubReader,
            template: Union[str, cv2.UMat], 
            readers: Dict[str, StrhubReader]) -> str:
    r"""
    
    .. note::
        Assumes input is already up-right. 
    
    """
    # * glob a rough area to work with
    img = img[600:800, 400:1200]

    # * Find student ID location
    locs = np.stack(template_match(img, template))
    x = locs.min(axis=0)[0]
    y = locs.min(axis=0)[1]
    h = int(locs.min(axis=0)[2] * 2.2) # use width of bracket as reference for height
    # make view larger
    x -= 5
    y -= int(h * 0.7)
    w = int(locs.max(axis=0)[2] * 17)
    bbox = (x, y, w, h)
    patch = img[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[2] + bbox[0]].copy()

    # * match template to remove all brackets
    # cover all brack with white rectangles
    locs = np.stack(template_match(patch, template, 0.7))
    w, h = locs[10, 2:]
    h = h * 49
    w += 10
    h -= 3
    
    locs = group_locations(locs[:, :2], min_radius=15)
    locs.sort(axis=0)

    if not len(locs) == 10:
        raise ArithmeticError(f"Error for {data_path}, got: {len(locs)}")
    
    digit_blocks = []
    for l in locs:
        x, y = l
        x = int(np.ceil(x)) - 10
        y = int(np.ceil(y)) - h + 5
        
        # off set
        x += 5
        y += 3
        digit_blocks.append(patch[y:y+h, x:x+w])
    
    # Resize all block to same shape
    digit_blocks_new = []
    for blk in digit_blocks:
        h, w =  digit_blocks[0].shape[:2]
        digit_blocks_new.append(cv2.resize(blk,(w, h)))

    new_patch = np.concatenate(digit_blocks_new, axis=1)
    
    # * post-process new-patch prior to ocr
    # create binary labels
    ret, new_patch_bin = cv2.threshold(new_patch, 200, maxval=255, type=0)
    # lable connected components
    nlabels, labels, cvstats, centroids = cv2.connectedComponentsWithStats(~new_patch_bin[...,0], None, None, None, 8, cv2.CV_32S)
    # filter from areas
    area = cvstats[1:, cv2.CC_STAT_AREA]
    mask = np.zeros_like(new_patch_bin).astype('uint8')
    for j in range(nlabels - 1):
        if centroids[j][1] > 45: # remove lower dots (0 is foreground)
            mask[labels == j + 1] = 255
    # mask original image 
    new_patch[mask != 0] = 255

    # * read text
    res = {}
    for name, reader in readers.items():
        sid = read_text(new_patch, reader)
        res[name] = sid
    return find_mode(res)