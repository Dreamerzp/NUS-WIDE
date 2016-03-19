from PIL import Image

import glob

count = 0
height_list = list()
width_list = list()

for file in glob.glob("/home/darshan/Documents/NUS-WIDE-IMAGE/*jpg"):

    count += 1
    parts = file.split("/")

    try:
        with Image.open(file) as image:
            new_image = image.resize((24, 24))
            new_image.save("/home/darshan/Documents/NUS-WIDE-SAMPLE/"+str(parts[-1]))
        print count
        if count == 500:
            break

    except Exception:
        print 'Exception Image IDs ' + str(parts[-1])

