from zipfile import ZipFile
zf = ZipFile('human.zip', 'r')
zf.extractall('./')
zf.close()

from torchvision import transforms
from PIL import Image
import os

dir = './human'

training_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.5)),
                transforms.ColorJitter(brightness=(1, 2.3), contrast=(0.5, 1.5), saturation=(0.5, 2), hue=(-0.1, 0.1)),
                transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), translate=(0.1,0.1)),
                ])

for a_file in os.listdir(dir):
  file_path = os.path.join(dir, a_file)
  img = Image.open(file_path)
  img1 = training_transforms(img)
  img1.save("./augmentated_images/{}.jpg".format(a_file))