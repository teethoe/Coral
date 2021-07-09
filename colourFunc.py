def bgr2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[2]), int(colour[1]), int(colour[0]))


def rgb2hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


def bgr2rgb(colour):
    return [colour[2], colour[1], colour[0]]


def normalise_hsv(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        norm[i][0] *= (1/180)
        for j in range(2):
            norm[i][j+1] *= (1/255)
    return norm


def normalise_rgb(arr):
    norm = arr.copy()
    for i in range(len(norm)):
        for j in range(3):
            norm[i][j] *= (1/255)
    return norm


def denormalise_hsv(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        arr[i][0] *= 180
        for j in range(2):
            arr[i][j+1] *= 255
    return arr


def denormalise_rgb(norm):
    arr = norm.copy()
    for i in range(len(arr)):
        for j in range(3):
            arr[i][j] *= 255
    return arr
