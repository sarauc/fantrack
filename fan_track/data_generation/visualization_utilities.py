import os

import cv2

from enum import Enum

def read_img(img_path, img_name):
    '''
        Read the current image.
        Inputs: img_path is the path to image files.
                img_name is the name of the current
                image of the scene.
        Return: rgb image of the current scene.
    '''

    # Load an color image without changing
    frame = cv2.imread(img_path + '/' + img_name, -1)

    # since OpenCV follows BGR order
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb_img

class colors(Enum):

    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)

def create_video(imgs_path):

    _, _, frames = next(os.walk(imgs_path))

    frames.sort()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    f = frames[0]

    img = cv2.imread(imgs_path + '/' + f)

    rows, cols, _ = img.shape

    video = cv2.VideoWriter('output.avi',fourcc, 20.0, (cols,rows))

    video.write(img)

    for f in frames[1:]:
        img = cv2.imread(imgs_path + '/' + f)

        video.write(img)


    video.release()

def draw_annotated_bbox(image,save_path,image_name,T):
    '''
        Draw annotated bbox around tracked objects
        on the image.
        Inputs: image where objects are located.
                save_path is the path where the
                image including bboxes is saved.
                image_name is the name of the
                saved image.
                T is the list of tracked objects.
    '''

    for t in T:

        # width and height
        w = t.x[2]
        h = t.x[3]

        # top-left and bottom-right corners
        x1 =  int(t.x[0])
        x2 =  int(t.x[0]+w)

        y1 = int(t.x[1])
        y2 = int(t.x[1]+h)

        if (t.is_tentative()):
            colour = colors.RED.value
        else:
            colour = colors.BLUE.value

        cv2.rectangle(image,(x1,y1),(x2,y2),colour,2)

        cv2.putText(img = image,
                    text = 'Id:' + str(t.trackId),
                    org = (x1+1,y1-1),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1.5,
                    color = colors.GREEN.value,
                    thickness = 2)


    cv2.imwrite(save_path + '/' + image_name, image)
