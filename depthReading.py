import slamBotHD as sb
from PIL import Image 
import numpy as np
import cv2
import time



sb.startUp()
sb.readCoreData()
   
RAD_PER_PIXEL = 0.001554434
RAD_OFFSET = 0.497418837

time.sleep(1)

def depth_to_centimetres(depth_array):
	centimetres_array = depth_array * 1000
	centimetres_array = centimetres_array.astype(np.uint16)

	return centimetres_array


def return_depth():

    elapsed_time = 0
    start_time = time.time()
    #depthArray = sb.imgDepth
    depthArray = depth_to_centimetres(sb.imgDepth)

    #Should take average of 3 scans, top mid below
    avgHorizontalAxis = (depthArray[239, :] + depthArray[240, :] + depthArray[241, :]) / 3
    
    
    center_pixel = np.mean(avgHorizontalAxis[318:322])
    #print(horizontalArrayAxis)

    #print("Middle distance: ", center_pixel)


    end_time = time.time()
    elapsed_time = end_time - start_time

    return avgHorizontalAxis

return_depth()

#img = Image.fromarray(return_depth()).show()



#timeStamp = sb.depthTimestamp * 10 ** -9
#print("Depth Time Stamp: ", timeStamp, " : ", elapsed_time)
#img = Image.fromarray(sb.imgColor).show()


#sb.shutDown()