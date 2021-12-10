# Pedestrian-Detection
Given an input image which may contain multiple pedestrians, a pedestrian detection system detects each person and returns a bounding box for each detection


This repo uses OpenCV to do template tracking using 2 methods:
* ***HoG Method:***
  - Firstly, we use a pretrained HoG detector (trained on the INRIA dataset as per the original paper [Histograms of Oriented Gradients for Human Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467360)). For this, we use OpenCV’s HOGDescriptor() with getDefaultPeopleDetector() [link](https://docs.opencv.org/4.5.3/d5/d33/structcv_1_1HOGDescriptor.html#a9c7a0b2aa72cf39b4b32b3eddea78203)

  - Next, we extract HoG features using [skimage's HoG feature extractor](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) and train our own SVM classifier on top of it. For this, prepare the training data with positive and negative samples for SVM.

* ***Faster-RCNN Method:*** Here, we use the pre-trained Faster-RCNN detector (as described in [Faster r-cnn: Towards real-time object detection with region proposal networks](https://arxiv.org/pdf/1506.01497.pdf)).
  - We implenmted PyTorch's implementation of Faster-RCNN. Here's a link to [PyTorch Faster-RCNN Tutorial](https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)
  - The pre-trained model detects multiple object categories, but we are only concerned with the ‘person’ category. Since we used a model trained on the COCO dataset, we only needed the predictions for class label 1, which corresponds to the ‘person’ category

### Dataset ###
The dataset used is "PennFudanPed" and can be downloaded from [here](https://www.cis.upenn.edu/~jshi/ped_html) and can be stored in PennFudanPed folder

### Using This Repo [TEMPLATE] ###
1. Download & put the data in their particular folders (like baseline,illumination, jitter & moving_bg)
2. Perform the background subtraction using:

  ` python main.py -i datasetname/input -o datasetname/result -c X -e datasetname/eval_frames.txt `,
  
  where i and o are paths for the input folder and target folder for predicted masks recpectively,
  eval frames.txt file contains the starting and ending frame that will be used for evaluation,
  and c is category name ("b" for baseline, "i" for illumination, "j" for camera jitter, and "d": for dynamic background)
  
  python eval.py --pred_path baseline/result --gt_path baseline/groundtruth
  
3. Evaluate the performance using:

    ` python eval.py --pred_path datasetname/result --gt_path datasetname/groundtruth`,
  
  where all the arguments are self-explanatory
