# LIBS
import cv2 as cv
import pandas as pd
import numpy as np
import math

# USER VARS
windowSize = (800, 800)
startLocation = (1000, 6000)  # WHERE BOT STARTS
dummyRes = (12000, 12000)  # RES IT WILL DRAW ON, MAKE VARIABLE SIZE LATER
csvLines = 2743
fovDeg = 57
fovWidthPixCount = 640

# GLOBAL VARS
filePath = "F:/DownloadsFolder/run_2_cleaned_mm_rads.csv"
data = pd.read_csv(filePath, header=None).to_numpy()
img = np.zeros((dummyRes[0], dummyRes[1], 1), np.uint8)
colourImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
deltaTheta = math.radians(fovDeg / fovWidthPixCount)  # CHANGE IN ANG FOR EACH PIXEL (RAD)
travelPath = []


#Need info from movement function of where the bot currently is
#in the form of [x, y, theta]
#Need info from depth reading in array


#Right now its doing some weird rounding for pixels, but I think it should be
#different for the real bots movements

def drawMap(botLocation, depthRead):
    x = botLocation[0]  # Current robot position: X
    y = botLocation[1]  # Y
    angRad = botLocation[2]  # Direction Robot is facing (Rads)


    depthRead[depthRead < 505] = 0  # NULL useless values (5mm cushion)
    depthRead[depthRead > 4995] = 0 #Scans below and above 0.5m are NaNs, replaced with 0 I guess
    currentLoc = (round(startLocation[0] - x), round(startLocation[1] - y))
    travelPath.append(currentLoc)

    for j in range(0, fovWidthPixCount):
        xPos = math.sin((math.radians(fovDeg / 2)) - angRad - (deltaTheta * j)) * depthRead[j]  # Get 'opposite'
        yPos = math.cos((math.radians(fovDeg / 2)) - angRad - (deltaTheta * j)) * depthRead[j]  # Get 'adjacent'
        depthReadLocation = (round(currentLoc[0] - yPos), round(currentLoc[1] + xPos))
        cv.circle(colourImg, depthReadLocation, 1, (255, 255, 255), -1)

    #This stuff is for showing the map updates iteratively

    # Draw traversed path
    cv.circle(colourImg, currentLoc, 12, (0, 0, 255), -1)

    # Display the updated image iteratively
    displayImage = cv.resize(colourImg.astype(np.uint8), windowSize, interpolation=cv.INTER_AREA)
    cv.imshow('Mapped Area', displayImage)












#Only useful for making it pretty
def rotate_image(image, angle):  # Rotates image, nothing special
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rotationMatrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotatedImage = cv.warpAffine(image, rotationMatrix, (width, height))
    return rotatedImage

#Only useful for making it pretty
def crop_image(image):  # Crops outside border (call after rotation)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    xMin, yMin = image.shape[1], image.shape[0]  # Init vars for box
    xMax, yMax = 0, 0
    for contour in contours:  # Compute box w/ everything inside
        x, y, w, h = cv.boundingRect(contour)
        xMin = min(xMin, x)
        yMin = min(yMin, y)
        xMax = max(xMax, x + w)
        yMax = max(yMax, y + h)
    croppedImage = image[yMin:yMax, xMin:xMax]
    return croppedImage






#Main loop (would be main function for bot I guess)
for i in range(0, csvLines):
    botLocation = data[i, 0:3]

    depthRead = data[i, 3:]
    drawMap(botLocation, depthRead)
    if cv.waitKey(1) == 27:  # Press 'Esc' to stop the process
        break








# Not sure what all this is, just stuff to clean, but I think it also detects the lines
# which is important but ill ignore it for now and clean up later

# Final processing after the loop
gray = cv.cvtColor(colourImg, cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY_INV)
binary = cv.bitwise_not(binary)
contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv.contourArea)
rect = cv.minAreaRect(contour)
(center, (width, height), angle) = rect
if width < height:
    angle = angle - 90

# Rotate and crop the image
rotatedImage = rotate_image(colourImg, -angle)
croppedImage = crop_image(rotatedImage)
gray = cv.cvtColor(croppedImage, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(gray)
_, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY_INV)

# Apply morphological operations
kernel = np.ones((5, 5), np.uint8)
closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
edges = cv.Canny(closed, 100, 150)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)

# Draw the detected lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(closed, (x1, y1), (x2, y2), (0, 0, 0), 16)  # Red color, 2 thickness
cleaned = cv.bitwise_not(closed)
cleaned = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)



# Final display
finalImage = cv.resize(cleaned.astype(np.uint8), windowSize, interpolation=cv.INTER_AREA)
cv.imshow('Final Mapped Area', finalImage)
cv.waitKey(0)
cv.destroyAllWindows()
