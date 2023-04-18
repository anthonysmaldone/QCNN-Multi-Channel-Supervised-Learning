# Import the necessary libraries
import random
from PIL import Image
from numpy import asarray
import numpy as np 
path ='/mnt/c/Users/Anthony M. Smaldone/Desktop/colors_shapes/'

#colors = ['brown.png']

colors = ['brown.png','brown_corner.png','brown_cross.png','brown_plus.png','orange.png','orange_corner.png','orange_cross.png','orange_plus.png','pink.png','pink_corner.png','pink_cross.png','pink_plus.png','purple.png','purple_corner.png','purple_cross.png','purple_plus.png']

#colors = ['blue.png','yellow.png','green.png']
for i in range(len(colors)):
  # load the image and convert into
  # numpy array
  numpydata_pure = np.asarray(Image.open(path+colors[i]).convert('RGB')).copy()
  #print(numpydata_pure.shape)
  # asarray() class is used to convert
  # PIL images into NumPy arrays
  #numpydata = asarray(img)
  #numpydata = numpydata/255

  noise_percent = 0.2
  noise_pixels = int(noise_percent*(len(numpydata_pure[0]))**2)
  N = 400
  print(colors[i])
  #print(numpydata_pure)
  for j in range(N):
    #print(j)
    numpydata = numpydata_pure.copy()
    k = 0
    while k < noise_pixels:
      rand_h = random.randint(0,len(numpydata[0])-1)
      rand_w = random.randint(0,len(numpydata[0])-1)
      #print(k)
      if numpydata[rand_h][rand_w][0] != 0 and numpydata[rand_h][rand_w][1] != 0 and numpydata[rand_h][rand_w][2] != 0:
        numpydata[rand_h][rand_w][0] = 0.0
        numpydata[rand_h][rand_w][1] = 0.0
        numpydata[rand_h][rand_w][2] = 0.0
        #print('written')
        k = k + 1
      #else:
       # print(numpydata[rand_h][rand_w][0])
        #print(numpydata[rand_h][rand_w][1])
        #print(numpydata[rand_h][rand_w][2])

    #print(numpydata.shape)
    img_noise = Image.fromarray(numpydata, 'RGB')
    img_noise = img_noise.save(path+"noisy_colors/"+str(i)+"/"+str(j)+".png")
