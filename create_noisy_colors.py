# Import the necessary libraries
import random
from PIL import Image
from numpy import asarray
import numpy as np 
import os

# the directory where the base colors are stored
paths = ['./mixed_colors/','./mixed_colors_shapes/']

for path in paths:
# get file names only
    colors = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    print(colors)
    os.mkdir(path+"noisy_colors/")
    for i in range(len(colors)):

# create the one hot encoded folders
      os.mkdir(path+"noisy_colors/"+str(i))

# load the image and convert into numpy array
# asarray() class is used to convert PIL images into NumPy arrays
      numpydata_pure = np.asarray(Image.open(path+colors[i]).convert('RGB')).copy()

# percent of pixels to corrupt to black
      noise_percent = 0.2
      noise_pixels = int(noise_percent*(len(numpydata_pure[0]))**2)

# number of total images to create
      N = 400
      print(colors[i])

      for j in range(N):

        numpydata = numpydata_pure.copy()
        k = 0
        while k < noise_pixels:

# find a random pixel
          rand_h = random.randint(0,len(numpydata[0])-1)
          rand_w = random.randint(0,len(numpydata[0])-1)

# check to see if pixel is already black, if not then blacken
          if numpydata[rand_h][rand_w][0] == 0 and numpydata[rand_h][rand_w][1] == 0 and numpydata[rand_h][rand_w][2] == 0:
            pass

          else:
            numpydata[rand_h][rand_w][0] = 0.0
            numpydata[rand_h][rand_w][1] = 0.0
            numpydata[rand_h][rand_w][2] = 0.0
            k = k + 1

#convert the array back to an image and save
        img_noise = Image.fromarray(numpydata, 'RGB')
        img_noise = img_noise.save(path+"noisy_colors/"+str(i)+"/"+str(j)+".png")
