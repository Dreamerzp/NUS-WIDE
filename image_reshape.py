from PIL import Image

import glob

count = 0
height_list = list()
width_list = list()
for file in glob.glob("/Users/Darshan/Documents/kaggle/NUS-WIDE_IMAGES/NUS-WIDE/*jpg"):
    count += 1
    parts = file.split("/")

    try:
        with Image.open(file) as image:
            new_image = image.resize((224, 224))
            new_image.save("/Users/Darshan/Documents/kaggle/NUS-WIDE_IMAGES/NUS-WIDE-RESIZE/"+str(parts[-1]))

        if count % 1000 == 0:
            print count
    except Exception:
        print 'Exception Image IDs' + str(parts[-1])

