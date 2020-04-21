from tkinter import Tk
from tkinter.filedialog import askopenfilename

def return_file_name():
    Tk().withdraw()
    filename = askopenfilename()
    return filename