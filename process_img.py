import os
import cv2 as cv
import numpy as np

# Read File in Path
def read_file(path):
    images = []
    for filename in sorted(os.listdir(path)):
        img = cv.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

def rotate(image, angle):
    height, width = image.shape[:2]
    rot_mat = cv.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv.warpAffine(image, rot_mat, (width,height))
    return rotated_img

# Apply Canny with Automatic Parameter
def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using computed median
    lower = int(max(0, (1.0-sigma) * v))
    upper = int(min(255, (1.0+sigma) * v))
    canny = cv.Canny(image, lower, upper)
    return canny

def preprocessing(image, name):
    # Grayscaling
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('./Result/gray/'+name, gray)

    # Equalize Histogram
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    hist = clahe.apply(gray)
    cv.imwrite('./Result/hist/'+name, hist)

    # Blur Image (Reduce Noise)
    blur = cv.medianBlur(hist, 5)
    cv.imwrite('./Result/blur/'+name, blur)

    # Mask Pupil
    _, thresh = cv.threshold(blur, 10, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxContour = 0
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    ## find enclosing circle of pupil contour
    (x,y) ,r = cv.minEnclosingCircle(maxContourData)
    center = (int(x), int(y))
    radius = int(r)

    img = blur.copy()
    masked_pupil = cv.circle(img, center, radius+10, (255,255,255), -1)
    cv.imwrite('./Result/masked_pupil/'+name, masked_pupil)
    return masked_pupil

def segmentation(image, name):
    center = (int(image.shape[0]/2),int(image.shape[0]/2))
    radius = int(image.shape[0]/2)

    # Convert to cartesian
    cartesian = cv.linearPolar(image, center, radius, cv.WARP_FILL_OUTLIERS)
    cartesian = rotate(cartesian, -90)
    cv.imwrite('./Result/cartesian/'+name, cartesian)

    # Crop Target
    [y, x] = cartesian.shape
    target = cartesian[0:int(y), 1:int(x/12)]
    cv.imwrite('./Result/target/'+name, target)

    # Crop Pupil Area
    _, thresh = cv.threshold(target, 250, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxContour = 0
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    rect = cv.boundingRect(maxContourData)
    x,y,w,h = rect
    crop_pupil = target[y+h:, 0:]
    cv.imwrite('./Result/crop_pupil/'+name, crop_pupil)

    # Specify ROI
    [y,x] = crop_pupil.shape
    roi = crop_pupil[0:int(y/2), 0:]
    cv.imwrite('./Result/roi/'+name, roi)

    #Resize ROI
    roi_res = cv.resize(roi, (50,50))
    cv.imwrite('./Result/roi_res/'+name, roi_res)
    return roi_res

def find_feat(image, name):
    canny = auto_canny(image)
    cv.imwrite('./Result/canny/0.5/'+name, canny)
    # Flatten Image Array
    feature = canny.flatten()
    return feature


# PROCESS IMAGE HANDLER
def process_image(path, label):
    # Read File on Path
    data = read_file(path)
    print('Folder {} Contains {} Images'.format(label, len(data)))

    features = []

    for n, file in enumerate(data):
        name = label+'_{}.JPG'.format(n)
        # Preprocessing Image in Data
        masked_pupil = preprocessing(file, name)
        # Segmenting Image ROI
        roi = segmentation(masked_pupil, name)
        # Find Image Feature
        feature = find_feat(roi, name)
        features.append(feature)
    return features
