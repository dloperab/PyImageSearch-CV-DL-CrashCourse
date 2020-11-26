# PyImageSearch CV/DL CrashCourse

Repository for **FREE** Computer Vision, Deep Learning and OpenCV Crash Course.

- **Course URL:** <https://www.pyimagesearch.com/free-opencv-computer-vision-deep-learning-crash-course/>

# Environment Configuration

The development environment configuration was based on the following guide [How to install TensorFlow 2.0 on Ubuntu](https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/) from PyImageSearch blog.

However, you can check the environment.yml or requirements.txt.

# Course

## Day 1: Face detection with OpenCV and Deep Learning

* **Link:** <https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/>
* **Folder:** 01-deep-learning-face-detection

**Commands used:**

* **Face detection with Images:**

    > *$ python detect_faces.py --image images/rooster.jpg --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel*

    > *$ python detect_faces.py --image images/iron_chic.jpg --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel*

* **Face detection with Webcam:**

    > *$ python detect_faces_video.py --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel*

## Day 2: OpenCV Tutorial: A Guide to Learn OpenCV

* **Link:** <https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/>
* **Folder:** 02-opencv-tutorial

**Commands used:**

* **OpenCV tutorial:**

> *$ python opencv_tutorial_01.py*

* **Counting objects:**

> *$ python opencv_tutorial_02.py --image images/tetris_blocks.png*

## Day 3: Document scanner

* **Link:** <https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/>
* **Folder:** 03-document-scanner

**Commands used:**

> *$ python scan.py --image images/page.jpg*

## Day 4: Bubble sheet multiple choice scanner and test grader using OMR

* **Link:** <https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/>
* **Folder:** 04-omr-test-grader

**Commands used:**

> *$ python test_grader.py --image images/test_01.png*

## Day 5: Ball Tracking with OpenCV

* **Link:** <https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/>
* **Folder:** 05-ball-tracking

**Commands used:**

* **Using Video:**

    > *$ python ball_tracking.py --video ball_tracking_example.mp4*

* **Using Webcam:**

    > *$ python ball_tracking.py* (**Note:** To see any results, you will need a green object with the same HSV color range was used in this demo)

## Day 6: Measuring size of objects in an image with OpenCV

* **Link:** <https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/>
* **Folder:** 06-size-of-objects

**Commands used:**

> *$ python object_size.py --image images/example_01.png --width 0.955*

> *$ python object_size.py --image images/example_02.png --width 0.955*

> *$ python object_size.py --image images/example_03.png --width 3.5*

## Day 8: Facial landmarks with dlib, OpenCV, and Python

* **Link:** <https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/>
* **Folder:** 08-facial_landmarks

**Commands used:**

> *$ python facial_landmarks.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --image images/example_01.jpg*

> *$ python facial_landmarks.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --image images/example_02.jpg*

> *$ python facial_landmarks.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --image images/example_03.jpg*

## Day 9: Eye blink detection with OpenCV, Python, and dlib

* **Link:** <https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/>
* **Folder:** 09-blink-detection

**Commands used:**

> *$ python detect_blinks.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --video videos/blink_detection_demo.mp4*

## Day 10: Drowsiness detection with OpenCV

* **Link:** <https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/>
* **Folder:** 10-detect_drowsiness

**Commands used:**

> *$ python detect_drowsiness.py --shape-predictor model/shape_predictor_68_face_landmarks.dat --alarm sounds/alarm.wav*

## Day 12: A simple neural network with Python and Keras

* **Link:** <https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/>
* **Folder:** 12-simple-neural-network

**Note:** Create a folder structure called **/kaggle_dogs_vs_cats/train**, download the training dataset [Kaggle-Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and put the images into **train** folder.

**Command used - Training:**

> *$ python simple_neural_network.py --dataset kaggle_dogs_vs_cats --model output/simple_neural_network.hdf5*

**Command used - Test:**

> *$ python test_network.py --model output/simple_neural_network.hdf5 --test-images test_images*

## Day 13: Deep Learning with OpenCV

* **Link:** <https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/>
* **Folder:** 13-deep-learning-opencv

**Commands used:**

> *$ python deep_learning_with_opencv.py --image images/jemma.png --prototxt model/bvlc_googlenet.prototxt --model model/bvlc_googlenet.caffemodel --labels model/synset_words.txt*

> *$ python deep_learning_with_opencv.py --image images/traffic_light.png --prototxt model/bvlc_googlenet.prototxt --model model/bvlc_googlenet.caffemodel --labels model/synset_words.txt*

> *$ python deep_learning_with_opencv.py --image images/eagle.png --prototxt model/bvlc_googlenet.prototxt --model model/bvlc_googlenet.caffemodel --labels model/synset_words.txt*

> *$ python deep_learning_with_opencv.py --image images/vending_machine.png --prototxt model/bvlc_googlenet.prototxt --model model/bvlc_googlenet.caffemodel --labels model/synset_words.txt*

## Day 14: How to (quickly) build a deep learning image dataset

* **Link:** <https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/>
* **Folder:** 14-search_bing_api

**Commands used:**

> *$ python search_bing_api.py --query "pokemon_class_to_search" --output dataset/pokemon_class_to_search*

## Day 15: Keras and Convolutional Neural Networks (CNNs)

* **Link:** <https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/>
* **Folder:** 15-cnn-keras

**Command used - Training:**

> *$ python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle*

**Command used - Testing:**

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png*

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/bulbasaur_plush.png*

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/mewtwo_toy.png*

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/pikachu_toy.png*

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/squirtle_plush.png*

> *$ python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_hidden.png*

## Day 16: Real-time object detection with deep learning and OpenCV

* **Link:** <https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/>
* **Folder:** 16-real-time-object-detection

**Commands used:**

> *$ python real_time_object_detection.py --prototxt model/MobileNetSSD_deploy.prototxt.txt --model model/MobileNetSSD_deploy.caffemodel*

---

**Credits to Adrian Rosebrock on <http://www.pyimagesearch.com>**