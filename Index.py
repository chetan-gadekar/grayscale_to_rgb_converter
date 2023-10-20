from tkinter import *
from PIL import Image, ImageTk

root = Tk()


def colour_image():
    import numpy as np
    import cv2

    # --------Model file paths--------#
    proto_file = "model\colorization_deploy_v2.prototxt"
    model_file = "model\colorization_release_v2.caffemodel"
    hull_pts = "model\pts_in_hull.npy"
    img_path =  img_input.get()+ ".jpeg"
    # --------------#--------------#

    # --------Reading the model params--------#
    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)
    # -----------------------------------#---------------------#

    # -----Reading and preprocessing image--------#
    img = cv2.imread(img_path)
    scaled = img.astype("float32")/ 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    # -----------------------------------#---------------------#

    # add the cluster centers as 1x1 `convolutions` to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    # -----------------------------------#---------------------#

    # we'll resize the image for the network
    resized = cv2.resize(lab_img, (224, 224))
    # split the L channel
    L = cv2.split(resized)[0]
    # mean subtraction
    L -= 50
    # -----------------------------------#---------------------#

    # predicting the ab channels from the input L channel

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Take the L channel from the image
    L = cv2.split(lab_img)[0]
    # Join the L channel with predicted ab channel
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Then convert the image from Lab to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # change the image to 0-255 range and convert it from float32 to int
    colorized = (255 * colorized).astype("uint8")

    # Let's resize the images and show them together
    img = cv2.resize(img, (640, 640))
    colorized = cv2.resize(colorized, (640, 640))

    result = cv2.hconcat([img, colorized])

    cv2.imshow("Grayscale -> Colour", result)

    cv2.waitKey(0)


root.title("black and white image convert into colourful format")
root.minsize(1000, 1000)
root.geometry("1400x500")
root.configure(bg="#0096DC")

p = Label(root, text='Black white image to colour image converter', fg='black', bg='#0096DC')
p.pack(pady=(50, 20))
p.config(font=('verdana', 25))

bi = Image.open('images/bl.png')
rs_img = bi.resize((350, 300))
bi = ImageTk.PhotoImage(rs_img)
bi_l = Label(root, image=bi, fg='black', bg='#0096DC')
bi_l.pack(side=LEFT)

# 0...................................................................................................................................................................................................................bi_l.pack(side=LEFT)

ai = Image.open('images/Color_bl.png')
as_img = ai.resize((350, 300))
ai = ImageTk.PhotoImage(as_img)
ai_l = Label(root, image=ai, fg='black', bg='#0096DC')
ai_l.pack(side=RIGHT)

img1 = Label(root, text='Enter your black and white image name', fg='black', bg='#0096DC')
img1.pack()
img1.config(font=('verdana', 20))
img_input = Entry(root, width=20)
img_input.pack(ipady=10, pady=(50, 15))

button = Button(root, text='UPLOAD', bg='white', fg='black', width=8, height=2, command=colour_image)
button.pack(pady=(10, 20))
button.config(font=('verdana', 10))
p1 = Label(root, text="BLACK IMAGE   -->    COLOR IMAGE", fg="black", bg='pink')
p1.pack(pady=(150, 30))
p1.config(font=('verdana', 25))

root.mainloop()