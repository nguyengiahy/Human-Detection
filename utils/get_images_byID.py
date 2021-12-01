import os
import shutil

imageDir = "../../images"

filenames = os.listdir(imageDir)

# extract = os.listdir(imageDir)[0].split("_")[:3]
# ID = "_".join(str)

ID = ["8b24f052-8b80-11eb-87f2-0242ac110003_1_613",
	  "8b24555c-8b80-11eb-87f2-0242ac110003_0_38",
	  "8b264682-8b80-11eb-87f2-0242ac110003_1_57",
	  "8b264682-8b80-11eb-87f2-0242ac110003_0_39",
	  "8b264682-8b80-11eb-87f2-0242ac110003_0_144",
	  "8b264682-8b80-11eb-87f2-0242ac110003_1_40",
	  "8b264682-8b80-11eb-87f2-0242ac110003_0_299",
	  "8b23db7c-8b80-11eb-87f2-0242ac110003_0_8",
	  "8b26d304-8b80-11eb-87f2-0242ac110003_0_41",
	  "8b27a8c4-8b80-11eb-87f2-0242ac110003_1_264"]

for i in ID:
	count = 0;
	for filename in filenames:
		if i in filename:
			imgPath = os.path.join(imageDir,filename)
			newPath = os.path.join("./images/non_human",filename)
			shutil.move(imgPath, newPath)
			count += 1
			if count == 50:
				break