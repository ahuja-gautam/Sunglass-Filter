# Sunglass-Filter
Very Basic sunglass filter using OpenCV and simple morphological operations

# How it Works
1)We detect facial landmarks of the person in the frame.

2)We then perform an Affine Transform so that 3 points on the sunglasses are mapped on to 3 landmarks on the face. I chose the first chin point, the last chin point, and the third point on the nose bridge to be mapped. 

Note:
The points in the glasses have to be hard coded and were a result of trial and error.

The facial landmark detection was done using [this library](https://github.com/ageitgey/face_recognition). Thanks to u/ageitgey for developing such an impressive library.

Best results are obtained when the person is directly facing the camera.

# Scope for improvement

Detecting pitch, yaw and roll (pose estimation) of the face to further morph the sunglasses to the person's face.

# How to use

Run ```python Filter.py``` in your command line. To change the filter, you will have to exit the program and run it again.
