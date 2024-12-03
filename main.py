""" This is a script to mimic harry potter invisible cloak via getting frame from webcam """

# open camera
# captures background image
# continuously processes new frame
# detect blue color in each frame
# create mask for blue areas
# replace blue area with background
# display result in real-time

from inv_cloak import start_cam

def main():
    start_cam()


main()

