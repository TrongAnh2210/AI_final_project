# _______________________________________________________________Import thư viện cần có _______________________________________________________________
import matplotlib.pyplot as plt
import keras_ocr
import cv2
from skimage import morphology, io
from skimage import img_as_float, img_as_ubyte
import skimage.io as io
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
# _______________________________________________________________Load the model_______________________________________________________________
model = tf.keras.models.load_model(r'D:\python\my_model.h5')
# Khai báo classname 
class_name = ['0','1','2','3','4','5','6','7','8','9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# _______________________________________________________________BUTTON GET IMAGE EVENT _______________________________________________________________
def open_file_dialog():
    # Hiển thị hộp thoại chọn tập tin ảnh
    global filepath 
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
    # Hiển thị đường dẫn của ảnh
    if filepath:
        #label_.config(text="Đường dẫn: " + filepath)
        # Đọc ảnh
        image = Image.open(filepath)
        # Thay đổi kích thước ảnh để phù hợp với nút nhấn
        image = image.resize((200, 200))
        # Tạo đối tượng ImageTk từ ảnh
        image_tk = ImageTk.PhotoImage(image)
        label_image.config(image=image_tk)
        label_image.image = image_tk
# _______________________________________________________________Preprocess function for the input image_______________________________________________________________
def preprocess_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    image = cv2.resize(image, (28, 28))  # Resize the image to (28, 28)
    image = image.reshape(1, 28, 28, 1)  # Reshape the image to match the expected input shape
    image_normalized = image.astype('float32') / 255.0  # Normalize pixel values
    return image_normalized
# _______________________________________________________________PREDICT FUNCTION_______________________________________________________________
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_name[predicted_class_index]
    return predicted_class
# _______________________________________________________________DETECT TEXT USING OCR_______________________________________________________________
def detect(path):
    image = keras_ocr.tools.read(path) # load Image 
    f = plt.figure()
    cropdrawn = []
    detector = keras_ocr.detection.Detector(weights='clovaai_general') # detecting 
    boxes = detector.detect(images=[image])[0] # drawbox arroung the text 
    drawn =  keras_ocr.tools.drawBoxes(image=image, boxes=boxes)
    for i in range(0,len(boxes)):
         # crop the image inside the box 
        cropdrawn.append(image[int(boxes[i][0][1]):int(boxes[i][2][1]), int(boxes[i][0][0]-10):int(boxes[i][2][0]),:])
    #plt.imshow(drawn) 
    #plt.show()
    return cropdrawn
# _______________________________________________________________PROCESSING BUTTON EVENT_______________________________________________________________
def processing(): 
    global string 
    string = "" # clear string after change another picture
    if filepath:    
        drawn=detect(filepath) # call dect func 
        for i in range(len(drawn)):
            #string = " "
            mang_phan_tu = [] # clear 
# _______________________________________________________PREPROCESSING BEFORE CROP CHARACETERS_______________________________________________________________  
            img_gray = cv2.cvtColor(drawn[i], cv2.COLOR_BGR2GRAY) 
            img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
            _, img_binary = cv2.threshold(img_blur, 180, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            inverted_image = cv2.bitwise_not(img_binary)
            img_erode = cv2.erode(inverted_image, (3,3))
            img_dilate = cv2.dilate(img_erode, (3,3))
            #cv2.imshow('abc',img_dilate)
            up_points = (1000, 1000)
            resized_up = cv2.resize(img_dilate, up_points, interpolation= cv2.INTER_LINEAR)
            resized_up_gray = cv2.resize(img_gray, up_points, interpolation= cv2.INTER_LINEAR)
            k = 0 
            mang_zero = []
            mang_ve = []
# ______________________________________________________________SEGMENTATION WHICH IS CHARACTER_______________________________________________________________ 
            while(k<len(resized_up[0])-1):             ############################################################################
                k+= 1                                  #
                count = 0                              #
                for iz in range(0,len(resized_up)) :   #    ADD ALL COLUMNS which have all pixels 0 
                    if(resized_up[iz][k]==0):          #
                        count += 1                     #
                    if( count == len(resized_up)-2):   #
                        mang_zero.append(k)            ###############################################################################
            #print(mang_zero)
            for ix in range(0,len(mang_zero)-1) :     ###############################################################################
                if(mang_zero[ix+1]-mang_zero[ix]>1):  #
                    mang_ve.append(mang_zero[ix])     #GET THAT POSITION 
                    mang_ve.append(mang_zero[ix+1])   #
                                                      ##############################################################################
# _______________________________________________________________________________________________________________________________________________________           
            for j in range(0, len(mang_ve)-1,1):
                # CROP IMAGE AT POSION GOT RECENTLY 
                resized_up = cv2.rectangle(resized_up, ( mang_ve[j],0), ( mang_ve[j+1],100), (255, 0, 255), 2)
                # SAVE IT IN MANG_PHAN_TU ARRAY 
                mang_phan_tu.append(resized_up_gray[:, mang_ve[j]:mang_ve[j + 1]])
                #cv2.imshow('dilated_image'+str(j),mang_phan_tu[j])
            for i in range (0,len(mang_phan_tu),2):
                # CHOOSE IMAGE WHICH HAVE CHARACTER AND RESIZE IT BEFORE PREDICT 
                mang_phan_tu[i] = cv2.resize(mang_phan_tu[i], (28,28), interpolation=cv2.INTER_AREA)
#_________________________________________________________PROCESSING BEFORE PREDICT__________________________________________________________________
                image_gray = cv2.GaussianBlur(mang_phan_tu[i], (3,3), 0)
                _, img_binary = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                inverted_image = cv2.bitwise_not(img_binary)
                img_erode = cv2.erode(inverted_image, (3,3))
                img_dilate = cv2.dilate(img_erode, (3,3))
                cv2.imshow('dilated_image'+str(i), img_dilate)
#_________________________________________________________PREDICTING__________________________________________________________________
                image_normalized = preprocess_image(img_dilate)
                predictions = model.predict(image_normalized)
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_name[predicted_class_index]
                string = string + str(predicted_class) 
            # SHOW PREDICTED ON ENTRYBOX 
            string = string + " "
            print(string)
            entry.set(string)
# __________________________________________________************TKINTER********** ________________________________________________________________________________________           
# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Giao diện nhận diện chữ viết tay")
window.geometry("800x800")
window.config(bg= 'gray')
label_window = tk.Label(window)
label_window.place( x = 0 , y= 0)
# Tạo nhãn để hiển thị đường dẫn của ảnh
image_window = Image.open("D:\python\window_edited.png")
# Thay đổi kích thước ảnh để phù hợp với nút nhấn
image_window  = image_window.resize((800, 800))
# Tạo đối tượng ImageTk từ ảnh
image_window  = ImageTk.PhotoImage(image_window)
label_window.config(image=image_window)
label_window.image = image_window
# Tạo nhãn để hiển thị đường dẫn của ảnh
# Tạo nút nhấn
button_select = tk.Button(window, text="Chọn ảnh", command=open_file_dialog,width=10, height=10,bg = 'blue', font= 20, bd =10 )
button_select.place(x = 100 , y= 200)
# Tạo nhãn để hiển thị ảnh
label_image = tk.Label(window)
label_image.place( x = 500 , y= 200)
# Tạo nút nhấn 2 để xử lý dữ liệu 
button_select_2 = tk.Button(window, text="Processing ", command=processing, width=10, height=10, bg= 'blue', font = 20 , bd =10)
button_select_2.place(x=100, y =500 ) 
# Tạo entrybox để thể hiện chuỗi đọc được 
entry = tk.StringVar()
Entry1 = tk.Entry(window, font= 13, textvariable= entry, bd = 10)
Entry1.place( x = 500 , y= 600)
label_image2 = tk.Label(window)
label_image2.place( x = 500 , y= 800)
# Chạy giao diện chương trình
window.mainloop()
# Bấm phím bất kỳ để đóng cửa sổ hiển thị hình
cv2.waitKey(0)  
#Giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình 
cv2.destroyAllWindows()