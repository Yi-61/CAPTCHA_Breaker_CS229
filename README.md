# CAPTCHA_Breaker_CS229
This is the repository for our Stanford CS229 class project, Fall 2017.

This project aims to recognize letter CAPTCHAs, done by Nathan Zhao, Yi Liu, and Yijun Jiang.

## Contents
I. Preprocessing and Data Generation:
1. database_to_pickle.py -- converts a directory containing CAPTCHA images into a pickle file containing arrays with the image data and their respective labels
2. load_database.py -- loads a pickle object
3. dev_constants -- global variables for the project

II. Single-letter CAPTCHA breaker:
1. t-SNE
2. K-means
3. Support vector machine
4. Convolutional neural network (CNN) for single-letter recognition
5. CNN with transfer learning (VGG19)

III. Four-letter CAPTCHA breaker:
1. "Moving Window" algorithm
2. Multi-CNN
