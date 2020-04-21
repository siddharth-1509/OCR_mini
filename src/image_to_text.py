import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
def image_2_text(path):
    img=Image.open(path)
    text=pytesseract.image_to_string(img)
    print(text)