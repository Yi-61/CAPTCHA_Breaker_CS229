from io import BytesIO
from captcha.image import ImageCaptcha
from random import randint
import os
import numpy as np
import pickle
import dev_constants

fontDir = dev_constants.MY_PROJECT_PATH + '/Fonts/'
outDir = dev_constants.MY_PROJECT_PATH + '/Simple single letter dataset/'

fontName = 'calibri'
image = ImageCaptcha(fonts = [os.path.join(fontDir, (fontName + '.ttf'))])

exampleNum = 50000

features = []
labels = []

for counter in range(exampleNum):
    label = randint(0, 25)
    letter = '  ' + chr(np.array(label) + ord('A'))

    img = image.generate_image(letter)
    img = img.crop((0, 0, 75, 60))
    imgBW = img.convert('1')
    rowSum = 60 - np.array(imgBW).sum(0)
    rowSum = rowSum / rowSum.sum()
    rowSum = rowSum ** 2
    rowSum = rowSum / rowSum.sum()
    center = (int) (rowSum.dot(np.arange(rowSum.shape[0])))

    img = img.crop((center-20, 0, center+20, 60))
    data = np.array(img)

    features.append(data)
    labels.append(label)

    if (counter+1) % 1000 == 0:
        print('Generated %6d out of %6d' % (counter+1, exampleNum))
    if counter < 100: #Output 100 generated sample images
        img.save(os.path.join(outDir, (letter+str(randint(1,1000))+'.png')), format='png')

features = np.array(features, dtype='uint8')
labels = np.array(labels, dtype='uint8')

print('Saving data to file...')
outName = str(exampleNum)+'_single_letter.p'
pickle.dump((features, labels), open(os.path.join(outDir, outName), 'wb'))
print('Done')
