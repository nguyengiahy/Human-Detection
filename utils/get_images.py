import os
import shutil
import random

imageDir = "../../../images"

#filenames = os.listdir(imageDir)
filenames = random.sample(os.listdir(imageDir), 30)


for filename in filenames:
	imgPath = os.path.join(imageDir,filename)
	newPath = os.path.join("./images/new_images",filename)
	shutil.move(imgPath, newPath)