# ### CODE INFORMATION ###
# Author: Stephen J Lasky
# Date: May 3 2018
# This code was written by Stephen J Lasky
# It's purpose is to sort the CIFAR-10 dataset by various metrics:
#   metric = 1: sum of all color channels
#   metric = 2: average color of the image, converted to HSV, in list format [r,g,b]
#   metric = 3: summation of magnitude of a canny edge detector
#   metric = 4: diversity sorting. starts with 1 class, up to 10 classes. (THIS DOES NOT WORK YET.)
#
# ### HOW TO RUN ### #
# In the current form, this script can be run AS-IS.
# It will run the fucntion: sample_run_small(2, "results", False)
#   parameter=2:              This will sort the images by color using HSV (metric 2). Feel free to try parameters 1, 2 or 3. 4 does not currently work.
#   parameter="results":      This will print a file named "results" that contains the sorted image indices.
#   parameter="False":        This indicates that the results file doesn't already exist. It's best to not change this parameter.
# Finally, it will show an image of the result. This is probably the best way to visualize the results of this program, so pay attention to this.
#
# PLEASE NOTE: This code requires the python CIFAR-10 dataset unzipped in a /data folder. Otherwise, there will be an error.
#
# ### HAVE FUN! ###


import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import random
import colorsys
import cv2
import bisect



# global variables defined here
FILES = ["data/cifar-10-batches-py/data_batch_1",
             "data/cifar-10-batches-py/data_batch_2",
             "data/cifar-10-batches-py/data_batch_3",
             "data/cifar-10-batches-py/data_batch_4",
             "data/cifar-10-batches-py/data_batch_5"]
MINI_BATCH_SIZE = 256
NUM_CLASSES = 10        # NOTE: CIFAR-10 stores labels in range 0-9
DATA_COUNT = 50000
TEST_ITER = 0


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def print_img(data, im_idx, w, h):
    im = return_img_array(data, im_idx, w, h)

    plt.imshow(im, interpolation='nearest')
    plt.show()

    print(im[:,:,0])

def return_img_array(data, im_idx, w, h):
    im = np.zeros((h, w, 3), dtype=np.uint8)
    offset = h * w

    pixel_idx = 0
    for row in range(0, h):
        for col in range(0, w):
            im[row, col, 0] = data[im_idx, pixel_idx]  # red channel
            im[row, col, 1] = data[im_idx, pixel_idx + offset]  # blue channel
            im[row, col, 2] = data[im_idx, pixel_idx + 2 * offset]  # green channel
            pixel_idx = pixel_idx + 1

    return im

# data_row is a ROW of data from the CIFAR data set. therefore, this only returns ONE image
def cifar_im_to_std_im(data_row, w, h):
    im = np.zeros((h, w, 3), dtype=np.uint8)
    offset = h * w

    pixel_idx = 0
    for row in range(0, h):
        for col in range(0, w):
            im[row, col, 0] = data_row[pixel_idx]                   # red channel
            im[row, col, 1] = data_row[pixel_idx + offset]          # blue channel
            im[row, col, 2] = data_row[pixel_idx + 2 * offset]      # green channel
            pixel_idx = pixel_idx + 1

    return im


# input: list of images that are stored in the height x width x channels matrix format
def combine_img(im_list):
    # first, compute the size of the canvas
    # we will make it the CEILING(sqrt(num_ims))
    num_im = len(im_list)
    canvas_size = int(math.ceil(math.sqrt(num_im)))

    # now, generate the canvas
    im_w = len(im_list[0])
    im_h = len(im_list[0][0,:])
    canvas_w = im_w  * canvas_size
    canvas_h = im_h * canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    im_num = 0
    for imy in range(0,canvas_size):
        for imx in range(0,canvas_size):
            canvas[im_h * imy : im_h * imy + im_h, im_w * imx : im_w * imx + im_w,] = im_list[im_num]
            im_num = im_num + 1
            if im_num >= num_im:
                break
        if im_num >= num_im:
            break

    return canvas
def print_img2(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()


# Returns a numerical score for this image. This is what we will sort the image based on.
#   im      numpy array that
# metric = 1: sum of all color channels
# metric = 2: average color of the image, converted to HSV, in list format [r,g,b]
# metric = 3: summation of magnitude of a canny edge detector
# metric = 4: diversity sorting. starts with 1 class, up to 10 classes. (THIS DOES NOT WORK YET.)
global score_im_struct
score_im_struct = None
def score_im(im, metric, label):
    score = 0

    # sum of all color channels
    if metric == 1:
        score = np.sum(im)

    # metric = 2: average color of the image, converted to HSV
    elif metric == 2:
        im_h = len(im)
        im_w = len(im[0])
        num_pixels = im_h * im_w
        im = np.sum(im, axis=0)
        im = np.sum(im, axis=0)

        # average by dividing by the number of pixels, normalize on 0-1 by dividing by 255
        r = float(im[0]) / (num_pixels * 255)
        g = float(im[1]) / (num_pixels * 255)
        b = float(im[2]) / (num_pixels * 255)
        # score = [r,g,b]

        score = colorsys.rgb_to_hsv(r,g,b)

        return score

    # metric = 3: summation of magnitude of canny edge detector
    elif metric == 3:
        im_canny = cv2.Canny(im, 50, 100)   # apply canny edge detector
        score = np.sum(im_canny)            # sum the total magnitude of the detected edges as the score

    # metric = 4: diversity sorting. starts with 1 class, up to 10 classes.
    elif metric == 4:
        # first make sure that score_im_struct is set
        global score_im_struct
        if score_im_struct == None:
            score_im_struct = {}                                                        # set it to a dictionary
            score_im_struct["num_minibatches"] = int(DATA_COUNT / MINI_BATCH_SIZE)      # keep track of the numnber of minibatches
            score_im_struct["minibatches"] = []                                         # keep track of the minibatches data

            # initialize the "need" dictionary
            minibatches_per_level = int(score_im_struct["num_minibatches"] / NUM_CLASSES)
            for i in range(0, score_im_struct["num_minibatches"]):
                curr_level = 1 + int(i / minibatches_per_level)                         # the 'level' is the number of unique minibatches here
                if curr_level > NUM_CLASSES:
                    curr_level = NUM_CLASSES
                curr_need = {}

                curr_need["total_items_in_batch"] = 0
                curr_need["num_unique_classes"] = 0
                curr_need["is_class_x_here"] = []
                add_one = 0
                if curr_level * int( math.ceil( MINI_BATCH_SIZE / curr_level ) ) < MINI_BATCH_SIZE:     # may need to add one to ensure we get to at least the minibatch size
                    add_one = 1
                for j in range(0, NUM_CLASSES):
                    curr_need[j] = int( math.ceil( MINI_BATCH_SIZE / curr_level ) ) + add_one
                    curr_need["is_class_x_here"].append(False)

                score_im_struct["minibatches"].append(curr_need)    # finally, add it


            # now, store a SEPERATE list of the "highest" needs
            score_im_struct["highest_needs"] = []
            for c in range(0, NUM_CLASSES):                                 # for each class
                highest_needs = []
                for mb in range(0, score_im_struct["num_minibatches"]):     # for each minibatch
                    need = score_im_struct["minibatches"][mb][c]        # for the minibatch mb at class c, the need is
                    highest_needs.append((need,mb))
                score_im_struct["highest_needs"].append(highest_needs)



            print "initialization done."    # TODO: remove debug

        # OTHERWISE: the data structure has already been initialized. we just need to find the highest needs
        # basic idea: go to highest_needs list, pop the first one off, and then reinsert to maintain integrity of highest_needs list
        # once highest_needs reaches 0, just completely remove it from the list
        else:
            # global TEST_ITER
            # print TEST_ITER
            # TEST_ITER = TEST_ITER + 1

            if len(score_im_struct["highest_needs"][label]) == 0:
                print "error about to occur"

            mb_tuple = score_im_struct["highest_needs"][label].pop(0)
            mb = mb_tuple[1]    # RECALL: TUPLES STORED AS (COUNT, MINIBATCH)

            minibatches_per_level = int(score_im_struct["num_minibatches"] / NUM_CLASSES)
            curr_level = 1 + int(mb / minibatches_per_level)  # the 'level' is the number of unique minibatches here
            if curr_level > NUM_CLASSES:
                curr_level = NUM_CLASSES

            # case 1: this is the expected case. nothing needs to be changed with other data structures, and so we update the variable
            if score_im_struct["minibatches"][mb]["num_unique_classes"] == curr_level:
                # maintain integrity of minibatch header information here
                score_im_struct["minibatches"][mb][label] = score_im_struct["minibatches"][mb][label] - 1   # update minibatch count

                # maintain integrity of highest_needs list here
                new_mb_tuple = (mb_tuple[0], mb_tuple[1] - 1)
                if new_mb_tuple[1] > 0:  # ensure that the need is still at least 0
                    insert_pos = custom_bisect(score_im_struct["highest_needs"][label], new_mb_tuple, 0)   # key is ZERO because tuples are (COUNT, MINIBATCH)
                    if len(score_im_struct["highest_needs"][label]) == 117:      # TODO: REMOVE DEUB
                        print "too long"
                    score_im_struct["highest_needs"][label].insert(insert_pos, new_mb_tuple)

            # case 2: this is the case where we need to change up the needs because a minibatch is about to take on a new label
            else:

                score_im_struct["minibatches"][mb][label] = score_im_struct["minibatches"][mb][label] - 1   # update minibatch count
                score_im_struct["minibatches"][mb]["is_class_x_here"][curr_level - 1] = True                                   # update here to maintain the class diversity in this batch

                # maintain integrity of highest_needs list here
                new_mb_tuple = (mb_tuple[0], mb_tuple[1] - 1)
                if new_mb_tuple[1] > 0:     # ensure that the need is still at least 0
                    insert_pos = custom_bisect(score_im_struct["highest_needs"][label], new_mb_tuple, 0)  # key is ZERO because tuples are (COUNT, MINIBATCH)
                    if len(score_im_struct["highest_needs"][label]) == 117:      # TODO: REMOVE DEUB
                        print "too long"
                    score_im_struct["highest_needs"][label].insert(insert_pos, new_mb_tuple)

                # NOW. we need to figure out where we need to pull out potential "needs" from
                maxiumum_number_of_unique_batch_appearances = int( math.ceil(float(curr_level) / NUM_CLASSES * minibatches_per_level ) )   # at THIS level


                # count the number of minibatch appearances at this level
                current_number_of_unique_batch_appearances = 0
                for i in range(0,minibatches_per_level):
                    if score_im_struct["minibatches"][(curr_level-1) * minibatches_per_level + i]["is_class_x_here"][curr_level - 1]:
                        current_number_of_unique_batch_appearances = current_number_of_unique_batch_appearances + 1

                # NOW, go through and remove potential class needs from other minibatches if we have exceeded our need here.
                highest_need_mb_to_remove = []
                if current_number_of_unique_batch_appearances == maxiumum_number_of_unique_batch_appearances:       # if we have reached the limit
                    for i in range(0,minibatches_per_level):                                                        # for every minibatch at this level
                        mb_idx = (curr_level - 1) * minibatches_per_level + i                                       # get the minibatch index for the data structure
                        if score_im_struct["minibatches"][mb_idx]["is_class_x_here"][curr_level-1] == False:         # prune references to this minibatch for thsi level
                            highest_need_mb_to_remove.append(mb_idx)

                    highest_need_mb_to_remove.sort()
                    j = 0   # j increment variable for the "for" loop below
                    while j < len(score_im_struct["highest_needs"][label]):
                        if len(highest_need_mb_to_remove) <= 0:
                            break
                        mb_idx = highest_need_mb_to_remove[0]

                        if j >= len(score_im_struct["highest_needs"][label]):
                            print "248: about to fail"

                        if score_im_struct["highest_needs"][label][j][1] == mb_idx:                                   # if this needs removed
                            highest_need_mb_to_remove.pop(0)
                            score_im_struct["minibatches"][mb_idx][curr_level] = 0                                      # set need to 0 in main data struct
                            score_im_struct["highest_needs"][label].pop(j)                                              # pop from highest needs.
                            j -= 1
                        j += 1

                # TODO: Below is causing a problem. We need to come back here.
                # elif current_number_of_unique_batch_appearances > maxiumum_number_of_unique_batch_appearances:
                #     print "How did this happen?"




            # FINALLY: At this point, we have now assigned an image to a mini-batch.
            score = mb * MINI_BATCH_SIZE + score_im_struct["minibatches"][mb]["total_items_in_batch"]
            score_im_struct["minibatches"][mb]["total_items_in_batch"] += 1

            # we now only need to remove all other referencers when the size of the minibatch reaches the goal
            if score_im_struct["minibatches"][mb]["total_items_in_batch"] == MINI_BATCH_SIZE:
                for l in range(0,NUM_CLASSES):
                    score_im_struct["minibatches"][mb][l] = 0                       # set all in the main struct to 0

                    for i in range(0, len(score_im_struct["highest_needs"][l])):   # now go through highest needs and set them to 0 as well
                        if i >= len(score_im_struct["highest_needs"][l]):          # leave if we've reached the end of highest needs
                            break
                        if len(score_im_struct["highest_needs"][l]) <= i:       # TODO: remove this
                            print "about to fail"
                        if score_im_struct["highest_needs"][l][i][1] == mb:        # same mb?
                            score_im_struct["highest_needs"][l].pop(i)             # then remove from highest needs



    else:
        return -1

    return score

def score_all_ims(files, metric, im_w, im_h):
    score_list = []

    # for every data batch
    for file in files:
        # bring data into memory
        data = unpickle(file)
        labels = data["labels"]     # store the labels
        data = data["data"]         # store the images

        # # TEST REGION DELETE #
        # return return_img_array(data, 0, im_w, im_h)
        # # END TEST REGION, DELETE #

        for i in range(0,len(data)):
            im = return_img_array(data, i, im_w, im_h)
            label = labels[i]
            score = score_im(im, metric, label)
            score_list.append(score)

    return score_list

# prints a list to a file, with each element seperated by a newline character \n
#   fname:      string of the file name and the file path
#   list:       list of data, can be any data that can be turned into a string
#   overwrite:  boolean argument, when true this function will overwrite files with the same name already in the directory
#   NOTE: overwrite not yet implemented
def print_list_to_file(fname, list, overwrite):
    f = open(fname, "w+")
    for element in list:
        element = str(element)
        f.write(element +  "\n")

# partial code: https://stackoverflow.com/questions/3277503/how-do-i-read-a-file-line-by-line-into-a-list
# converts a file filled with strings of integers, separated by newlines, into a list of integers
def file_to_int_list(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for i in range(0,len(content)):
        content[i] = int(content[i])

    return content

# sample run of the program below
def sample_run_small(metric, output_file_name, FROM_FILE):
    FILES = ["data/cifar-10-batches-py/data_batch_1"]
    im_scores = -1

    # if we need to score and sort the indices from scratch
    if not FROM_FILE:
        im_scores = score_all_ims(FILES, metric, 32, 32)                            # score each and every image
        sorted_idxs = sort_scores(im_scores)                                        # finally get the sorted list of indices
        print_list_to_file(output_file_name + "_sorted_idx", sorted_idxs, False)    # print the sorted indexes to a file
        print_list_to_file(output_file_name + "_scores", sorted_idxs, False)  # print the sorted indexes to a file

    # if we are already reading in the scores from a precomputed file file
    else:
        sorted_idxs = file_to_int_list(output_file_name + "_sorted_idx")

    # create a canvas for display
    sorted_ims = get_ims(FILES, 10000, sorted_idxs, 32, 32)
    sorted_ims_list = []
    for i in range(0,len(sorted_ims)):
        if i % 10 == 0:
            sorted_ims_list.append(cifar_im_to_std_im(sorted_ims[i], 32, 32))

    canvas = combine_img(sorted_ims_list)
    print_img2(canvas)
def sample_run_full(metric, output_file_name, FROM_FILE):
    FILES = ["data/cifar-10-batches-py/data_batch_1",
             "data/cifar-10-batches-py/data_batch_2",
             "data/cifar-10-batches-py/data_batch_3",
             "data/cifar-10-batches-py/data_batch_4",
             "data/cifar-10-batches-py/data_batch_5"]
    im_scores = -1

    # if we need to score and sort the indices from scratch
    if not FROM_FILE:
        im_scores = score_all_ims(FILES, metric, 32, 32)                            # score each and every image
        sorted_idxs = sort_scores(im_scores)                                        # finally get the sorted list of indices
        print_list_to_file(output_file_name + "_sorted_idx", sorted_idxs, False)    # print the sorted indexes to a file
        print_list_to_file(output_file_name + "_scores", sorted_idxs, False)  # print the sorted indexes to a file

    # if we are already reading in the scores from a precomputed file file
    else:
        sorted_idxs = file_to_int_list(output_file_name + "_sorted_idx")

    # create a canvas for display
    sorted_ims = get_ims(FILES, 10000, sorted_idxs, 32, 32)
    sorted_ims_list = []
    for i in range(0,len(sorted_ims)):
        if i % 1000 == 0:
            sorted_ims_list.append(cifar_im_to_std_im(sorted_ims[i], 32, 32))

    canvas = combine_img(sorted_ims_list)
    print_img2(canvas)

# gets the histogram of the list of data
def get_hist(l, num_bins):
    plt.hist(np.asarray(l), num_bins)
    plt.show()

# sorts the socre of the list
def sort_scores(l):
    num = len(l)

    # attach tuples to remember index, 0 is score, 1 is index
    for i in range(0,num):
        l[i] = (l[i],i)

    # sort the tupled list by the score
    l.sort(key=lambda x: x[0])

    # remove the tuple for list return
    for i in range(0, num):
        l[i] = l[i][1]

    return l

# maintains order of list
# returns in cifar10 data format
def get_ims(fnames, num_per_file, indices, im_w, im_h):

    # set up return matrix
    num_ims = len(indices)
    return_data = np.zeros((num_ims, im_w * im_h * 3), dtype=np.uint8)

    for fname in fnames:
        # get data
        data = unpickle(fname)
        data = data["data"]

        # go through every image index in the list, testing if it is vaild for this particular file
        for i in range(0,num_ims):
            # if this image is in the current file
            if indices[i] >= 0 and indices[i] < num_per_file:
                return_data[i] = data[indices[i]]


            # subtract so that we don't consider this index again, as well as "refreshing" future indices
            indices[i] = indices[i] - num_per_file

    return return_data  # done

# generates a color rainbow image in numpy format
def generate_color_rainbow(mode):
    height = 128
    width = 8000

    im = np.zeros((height, width, 3), dtype=np.uint8)

    if mode == 0:
        pixel_step = int(256 * 256 * 256 / width)
        for i in range(0,width):
            color = dec_to_rgb(i * pixel_step)
            im[:,i,0] = color[0]    # r
            im[:, i, 1] = color[1]  # g
            im[:, i, 2] = color[2]  # b
    elif mode == 1:
        rgb_scheme = gen_new_rgb_scheme()

        pixel_step = int(256 * 256 * 256 / width)

        for i in range(0, width):
            if i * pixel_step > len(rgb_scheme):
                print(i*pixel_step, " bigger than", len(rgb_scheme))
                break
            color = rgb_scheme[i * pixel_step]
            im[:,i,0] = color[0]    # r
            im[:, i, 1] = color[1]  # g
            im[:, i, 2] = color[2]  # b

    # do a 256 x 256 platter of only 2 color channels
    elif mode == 2:
        im = np.zeros((256, 256, 3), dtype=np.uint8)

        for i in range(0,256):
            for j in range(0,256):
                im[i, j, 1] = i  # r
                im[i, j, 2] =  j  # g
                # im[:, i, 1] = i*j  # b




    return im

# returns tripe tuple (r,g,b) of the RBG value of a decimal number
def dec_to_rgb(dec):
    dec = int(dec)

    R_DIV = int(65536)
    G_DIV = int(256)
    B_DIV = int(1)

    r = 0
    g = 0
    b = 0

    # first compute R
    if dec / R_DIV > 0:
        r = dec / R_DIV
        dec = dec - r * R_DIV

    # now compute G
    if dec / G_DIV > 0:
        g = dec / G_DIV
        dec = dec - g * G_DIV

    # finally, the remainder SHOULDbe B
    b = dec

    return (r,g,b)

def gen_new_rgb_scheme():
    min = 0 # inclusive
    max = 255 # inclusive

    r = 0
    g = 0
    b = -1

    r_inc = True
    g_inc = True
    b_inc = True

    color_list = []

    while r + g + b != max * 3:
        # scenario 1: b needs increased
        if b_inc:
            b = b + 1
            if b > max:
                b = max
                b_inc = not b_inc

                # scenario 1.1: g needs to be increased
                if g_inc:
                    g = g + 1
                    if g > max:
                        g = max
                        g_inc = not g_inc
                        r = r + 1

                # scenario 1.2: g needs to be decreased
                else:
                    g = g - 1
                    if g < 0:
                        g = 0
                        g_inc = not g_inc
                        r = r + 1
        # scenario 2: b needs decreased
        else:
            b = b - 1
            if b < 0:
                b = 0
                b_inc = not b_inc

                # scenario 2.1: g needs to be increased
                if g_inc:
                    g = g + 1
                    if g > max:
                        g = max
                        g_inc = not g_inc
                        r = r + 1

                # scenario 2.2: g needs to be decreased
                else:
                    g = g - 1
                    if g < 0:
                        g = 0
                        g_inc = not g_inc
                        r = r + 1
        color_list.append((r,g,b))
    return color_list

# returns triple tuple (r,g,b), which are integers between [0-255] (inclusive)
def generate_random_rgb_colors(num):
    color_list = []

    for i in range(0,num):
        new_color = [random.random(), random.random(),random.random()]
        new_color[0] = int(round(new_color[0] * 255))
        new_color[1] = int(round(new_color[1] * 255))
        new_color[2] = int(round(new_color[2] * 255))

        color_list.append(tuple(new_color))

    return color_list

# input: takes a list of colors
# output: none, but displays the image of the color line in the order in which the colors were given
def print_color_set(color_set):
    num_colors = len(color_set)
    HEIGHT = 100

    im = np.zeros((HEIGHT, num_colors, 3), dtype = np.uint8)

    for i in range(0,num_colors):
        im[:, i, 0] = color_set[i][0]
        im[:, i, 1] = color_set[i][1]
        im[:, i, 2] = color_set[i][2]

    print_img2(im)

# custom bisect designed by Stephen Lasky. This function returns the index that a value x should be inserted at to maintain the sorted nature of the list
# the key value allows tuples and what not to be sorted
def custom_bisect(list, x, key):
    lo = 0
    hi = len(list) - 1
    probe = int(hi / 2)
    probe_old = None
    x = x[key]

    # handle edge cases
    if len(list) == 0:
        return 0
    elif x < list[0][key]:
        return 0
    elif x > list[len(list) - 1][key]:
        return len(list)


    while probe != probe_old:
        # print so that we can see whats going on
        # print_str = ""
        # for i in range(0,len(list)):
        #     if i == probe:
        #         print_str = print_str + "P"
        #     elif i == lo:
        #         print_str = print_str + "L"
        #     elif i == hi:
        #         print_str = print_str + "H"
        #     else:
        #         print_str = print_str + "."
        # # print print_str

        # meat of the search
        if list[probe][key] < x:
            lo = probe
        elif list[probe][key] > x:
            hi = probe
        elif list[probe][key] == x:
            return probe

        probe_old = probe
        probe = lo + int((hi-lo) / 2)

    return probe + 1

def metric4_test():
    data_idx_remain = []
    data_count = []
    METRIC = 4
    IMAGE = []  # fake image

    scores = []

    # generate fake labels
    for i in range(0,NUM_CLASSES):
        data_idx_remain.append(i)
        data_count.append(int(DATA_COUNT / NUM_CLASSES))

    # begin running the test and scoring images
    for i in range(0,DATA_COUNT):
        # generate the label for this iteration
        label = data_idx_remain[random.randint(0,len(data_idx_remain)-1)]
        data_count[label] -= 1
        if data_count[label] == 0:
            data_idx_remain.remove(label)

        # call score_img,
        global score_im_struct
        if i > 0:
            if len(score_im_struct["highest_needs"][label]) == 0:
                # print "error about to occur"        # TODO: REMOVE
                # print "error is at label:", label   # TODO: REMOVE
                continue
            if i % 5000 == 0:
                print "a 5000 iter..."              # TODO: REMOVE

        # if label == 9:
        #     continue    # skip label 9
        score = score_im(IMAGE, METRIC, label)
        scores.append((score,label))                # SCORES stored as (SCORE,LABEL)

    # finally, construct a visual representation:
    canvas = np.zeros((200, DATA_COUNT, 3), dtype=np.uint8)
    for i in range(0,DATA_COUNT):
        if i >= len(scores):
            break
        location = scores[i][0]
        label = scores[i][1]
        color = get_distinct_color(label)
        # print "writing", color, "to", location
        canvas[:, location, 0] = color[0]
        canvas[:, location, 1] = color[1]
        canvas[:, location, 2] = color[2]

    # print_img2(canvas)
    # matplotlib.image.imsave('m4_distribution.png', canvas)
    canvas2 = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas2[:,:,:] = canvas[:][:][0:200]
    # plt.imshow(canvas)
    # plt.savefig("m4_distribution2.png")
    matplotlib.image.imsave('m4_distribution.png', canvas2)

    print "Metric 4 test complete."

# this function returns a hand-coded distinct color generator
def get_distinct_color(index):
    r = 0
    g = 0
    b = -1


    colors = []
    for i in range(0,27):
        b += 1

        if b > 2:
            b = 0
            g += 1
            if g > 2:
                g = 0
                r += 1
        colors.append((r,g,b))

    colors.pop(len(colors)-1)   # remove white
    colors.pop(0)               # remove black
    return colors[index]


# TESTING GOES DOWN HERE #
sample_run_small(2, "results", False)
