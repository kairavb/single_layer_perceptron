import cv2 as cv
from keyboard import is_pressed
from os import remove, listdir
import numpy as np
from time import sleep


def input_neurons(img_pth):  # creating input neuron matrix/layer
    _ = cv.imread(img_pth, int(COLOR))
    _ = np.ravel(_)
    _ = np.zeros(RESIZE ** 2 * rgb) + _
    _ = _ / 255
    return _


def relu(z):
    return max(z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gather_data():
    k = [i * 0 for i in range(len(n_of_output_neurons))]
    while True:
        result, image = cam.read()  # reads/updates image
        cv.imshow("frame", image)  # shows image
        cv.waitKey(1)
        image = cv.resize(image, (RESIZE, RESIZE))  # resize image
        if is_pressed('a'):
            cv.imwrite("photo/{0}{1}.png".format(0, k[0]), image)  # writes image
            k[0] += 1
        if is_pressed('s'):
            cv.imwrite("photo/{0}{1}.png".format(1, k[1]), image)  # writes image
            k[1] += 1
        if is_pressed('d'):
            cv.imwrite("photo/{0}{1}.png".format(2, k[2]), image)  # writes image
            k[2] += 1
        if is_pressed('f'):
            cv.imwrite("photo/{0}{1}.png".format(3, k[3]), image)  # writes image
            k[3] += 1
        if is_pressed('esc'):
            break
    print('---------', k, '---------')
    cv.destroyAllWindows()


def train_weights():
    photos_list = [f for f in listdir('photo')]
    np.random.shuffle(photos_list)
    print('training...')
    for file in photos_list:
        label = int(file[0])
        m1 = input_neurons("photo/{0}".format(file))
        output = np.dot(m1, m2)

        for i in range(len(output)):
            output[i] = relu(output[i])

        if output[label] < bias:  # adding activation in desired weights
            m2.transpose()[label] += m1

        for r in n_of_output_neurons:  # subtracting activation from undesired weights
            if output[r] > bias and r != label:
                # if r != label:
                m2.transpose()[r] -= m1

        remove("photo/{0}".format(file))
    print('--------DONE!!!--------')


def see_weights():
    for i in n_of_output_neurons:
        x = m2.transpose()[i]
        x = x * 255
        x = x.reshape(RESIZE, RESIZE, rgb)
        cv.imwrite("weights/{0}.png".format(i), x)
    print('Successfully Exported Weights!!')


def test_model():
    while True:
        result, image = cam.read()  # reads/updates image
        image = cv.resize(image, (RESIZE, RESIZE))  # resize image to
        cv.imwrite("weights/x.png", image)  # writes image

        m1 = input_neurons("weights/x.png")
        output = np.dot(m1, m2)

        for i in range(len(output)):
            output[i] = int(relu(output[i]))
        print(output)

        try:
            remove("weights/x.png")  # deletes image
        except PermissionError:
            sleep(0.5)
            remove("weights/x.png")
        if is_pressed('esc'):
            break


cam = cv.VideoCapture(0, cv.CAP_DSHOW)  # sets camera
bias = 100  # same bias for all output neurons
RESIZE = 256
COLOR = True
if COLOR:
    rgb = 3
else:
    rgb = 1

# options = [i for i in range(12)]
# h1 = np.array([np.zeros(RESIZE ** 2 * rgb) for i in options]).transpose()

n_of_output_neurons = [i for i in range(4)]
m2 = np.array([np.zeros(RESIZE ** 2 * rgb) for i in n_of_output_neurons]).transpose()

while True:
    if is_pressed('q'):
        gather_data()
        sleep(1)

    if is_pressed('z'):
        train_weights()
        sleep(1)

    if is_pressed('m'):
        test_model()
        sleep(1)

    if is_pressed('p'):
        see_weights()
        sleep(1)

    if is_pressed('esc'):
        break
