from exceptions import IOError
import main

imagefile = raw_input("Enter the path for the target image! ")

output_size = raw_input("Enter the size of the output image Pixel x Pixel: ").strip().split('x')

try:
    window_size = int(raw_input("Enter size of tile: "))
except Exception as e:
    print str(e)

height = int(output_size[0])
width = int(output_size[1])

main.start(imagefile, height, width, window_size)
