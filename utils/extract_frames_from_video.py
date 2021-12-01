# import the necessary packages
import numpy as np
import cv2
import os

video_name = "vid31"
vs = cv2.VideoCapture(video_name + ".mp4")
read = 0
saved = 0
skip_rate = 5

# loop over frames from the video file stream
while True:
	# grab the frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break
	# increment the total number of frames read thus far
	read += 1
	# check to see if we should process this frame
	if read % skip_rate != 0:
		continue

	# write the frame to disk
	p = os.path.sep.join([video_name,"{}.png".format(video_name + "_" + str(saved))])
	cv2.imwrite(p, frame)
	saved += 1
	
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()