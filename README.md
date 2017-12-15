# CAPTCHA_Breaker_CS229
This is the repository for our Stanford CS229 class project, Fall 2017.

This project aims to recognize letter CAPTCHAs, done by Nathan Zhao, Yi Liu, and Yijun Jiang.

## Contents
I. CAPTCHA generator

II. Preprocessing:
database_to_pickle.py -- converts a directory of images into a pickle object containing the data and respective labels in numpy arrays
load_database.py
dev_constants.py -- global variables


III. Single-letter CAPTCHA breaker:
1. t-SNE
2. K-means
3. Support vector machine
4. Convolutional neural network (CNN) for single-letter recognition
5. CNN with transfer learning (VGG19)

IV. Four-letter CAPTCHA breaker:
1. "Moving Window" algorithm
2. Multi-CNN
