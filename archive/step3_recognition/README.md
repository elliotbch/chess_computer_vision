# How to use

## Directory
Once you have put our "recognition.py" script in your current directory, create a "frames" directory.

## Path to the test video
Then go to our script and modify the line 412.

You need to associate to the variable "video_moving" the absolute path between your current directory and the test video.

## Launch the script
You can then launch the script using the following command:

python recognition.py

## Results
A file called "states_{timestamp}.txt" will appear in your current directory.

It contains the game state evolution.

The "frames" directory will contains the drawed frames, using the format "image_frameid.png".