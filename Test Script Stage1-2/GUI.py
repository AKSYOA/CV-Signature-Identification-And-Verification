from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from joblib import dump, load

import script

root = Tk()
root.title("Signature Identification and Verification")
root.geometry("600x600")

root.filename = filedialog.askopenfilename(title="Select a file")
signatureImage = ImageTk.PhotoImage(Image.open(root.filename))
imageLabel = Label(image=signatureImage).pack()


def run():
    image = [root.filename]
    person_label = script.Identify(image, stage1_value.get())
    Label(root, text=person_label).pack()


stage1_value = IntVar()
Stage1_label = Label(root, text='Stage 1').pack()

hog_radio_button = Radiobutton(root, text="HOG", variable=stage1_value, value=1)
hog_radio_button.pack()

cnn_radio_button = Radiobutton(root, text="CNN", variable=stage1_value, value=2)
cnn_radio_button.pack()

stage2_value = IntVar()
Stage2_label = Label(root, text='Stage2').pack()

bow_radio_button = Radiobutton(root, text="BOW", variable=stage2_value, value=1)
bow_radio_button.pack()

siamese_radio_button = Radiobutton(root, text="SIAM", variable=stage2_value, value=2)
siamese_radio_button.pack()

ai_magic_button = Button(root, text='Ai Magic', width=10, command=run).pack()

mainloop()
