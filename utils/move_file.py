import os, shutil
source = "./source"
for file in os.listdir(source):
	file_path = os.path.join(source, file)
	shutil.copy(file_path, "./destination")