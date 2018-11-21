# PyImageSearch CV/DL CrashCourse

Repository for PyImageSearch Crash Course on Computer Vision and Deep Learning

* URL to course: <https://www.pyimagesearch.com/welcome-crash-course/>

## Day 1: Face detection with OpenCV and Deep Learning

* **Link:** <https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/>
* **Folder:** 01-deep-learning-face-detection

**Commands used:**

* **Object detection with Images:**

    > *$ python detect_faces.py --image images/rooster.jpg --prototxt model/deploy.prototxt.txt --model model/res10_300x300_ssd_iter_140000.caffemodel*

* **Object detection with Webcam:**

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

---

**Credits to Adrian Rosebrock on <http://www.pyimagesearch.com>**