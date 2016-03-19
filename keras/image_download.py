import urllib
import ssl
import urllib2
import time

#ctx = ssl.create_default_context()
#ctx.check_hostname = False
#ctx.verify_mode = ssl.CERT_NONE

photo_unavailable = "photo_unavailable"


def get_redirected_url(url):
    final_url = urllib2.urlopen(urllib2.Request(url)).geturl()
    return final_url


def download_images():

    image_not_available_lists = list()

    with open("/home/darshan/Documents/NUS-WIDE/image_exception.txt", "w") as image_exception:

        with open("/home/darshan/Documents/NUS-WIDE/NUS-WIDE-urls.txt", "r") as url_file:

            first_line = url_file.readline()
            #Ignore the first line

            url_large = 0
            url_middle = 0
            url_small = 0
            url_original = 0

            count = 0
            for line in url_file:

                image_name = None

                if count >= 0:
                    try:
                        parts = line.split(' ')

                        nbr = 0

                        for part in parts:
                            part = part.strip()
                            if part != '':
                                if part != 'null':
                                    if nbr == 1:
                                        image_name = part
                                    if nbr == 2:
                                        url_large += 1
                                    elif nbr == 3:
                                        url_middle += 1
                                    elif nbr == 4:
                                        redirected_url = get_redirected_url(part)
                                        #if photo_unavailable not in redirected_url:
                                            #urllib.urlretrieve(redirected_url, image_name + ".jpg", context=ctx)
                                        url_small += 1
                                        #else:
                                        #    print "Image unavailable"
                                        #    image_not_available_lists.append(image_name)
                                    elif nbr == 5:
                                        url_original += 1
                                nbr += 1

                    except Exception as e:
                        image_exception.write(image_name+"\n")
                        print e
                count += 1

    print("Total Links " + str(count))
    print("Large count : " + str(url_large))
    print("Middle count : " + str(url_middle))
    print("small count : " + str(url_small))
    print("original count : " + str(url_original))


def save_image_ids():

    image_ids = list()

    with open("../data/NUS-WIDE-urls.txt", "r") as url_file:

        first_line = url_file.readline()
        #Ignore the first line

        count = 0

        for line in url_file:

            image_name = None

            parts = line.split(' ')
            nbr = 0
            for part in parts:
                part = part.strip()
                if part != '':
                    if part != 'null':
                        if nbr == 1:
                            image_name = part
                            image_ids.append(image_name)
                            break
                    nbr += 1
            count += 1

    with open("../data/full_image_id.txt", "w") as file_writer:

        for image_id in image_ids:
            file_writer.write(image_id + "\n")

    print "Number of record loaded are {0}".format(count)

if __name__ == "__main__":
    save_image_ids()






