from PIL import Image  # used for loading images
import numpy as np
import os  # used for navigating to image path
import imageio  # used for writing images


# to change name from label to breed with a serial no.
if __name__=='__main__':
    i = 301
    for img in os.listdir('corona'):
        imgName = img.split('.')[0]  # converts '0913209.jpg' --> '0913209'
        label = 'Corona'
        path = os.path.join('corona', img)
        saveName = './c/' + label + ' image_'+ str(i) + '.jpg'
        image_data = np.array(Image.open(path).convert('RGB'))
        imageio.imwrite(saveName, image_data)
        i +=1