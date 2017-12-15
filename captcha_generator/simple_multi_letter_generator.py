from io import BytesIO
from captcha.image import ImageCaptcha
from random import randint
import os
import numpy as np
import pickle
import dev_constants

fontDir = dev_constants.MY_PROJECT_PATH + '\\Fonts\\'
outDir = dev_constants.MY_PROJECT_PATH + ('\\Simple ' + str(stringLen) + ' letter dataset\\')

fontName = 'calibri'
image = ImageCaptcha(fonts = [os.path.join(fontDir, (fontName + '.ttf'))])

separate = False
stringLen = 4
exampleNum = 5000

features = []
labels = []

for counter in range(exampleNum):
    label = []
    for i in range(stringLen):
        label.append(randint(0, 25))
    asc = np.array(label) + ord('A')
    string = ''.join(chr(i) for i in asc)

    if separate:
        modifiedString = ' ' + ' '.join(list(string))
    else:
        modifiedString = '  ' + string
    img = image.generate_image(modifiedString)
    img = img.crop((0, 0, 160, 60))
    data = np.array(img)

    features.append(data)
    labels.append(label)

    if (counter+1) % 1000 == 0:
        print('Generated %6d out of %6d' % (counter+1, exampleNum))
    # if counter < 10:
    #     img.save(os.path.join(outDir, (string+'.png')), format='png')

features = np.array(features, dtype='uint8')
labels = np.array(labels, dtype='uint8')

print('Saving data to file...')
if separate:
    outName = str(exampleNum) + '_' + str(stringLen) + '_letters_with_space.p'
else:
    outName = str(exampleNum) + '_' + str(stringLen) + '_letters_no_space.p'
pickle.dump((features, labels), open(os.path.join(outDir, outName), 'wb'))
print('Done')
