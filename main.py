import keras
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2 as cv


def display_on_board(main_arr, secondary_arr, main_img):
    for k in range(0,9):
        for j in range(0,9):
            if(secondary_arr[k][j]==0):
                cv.putText(main_img, str(main_arr[k][j]), (j*58+10, k*58 + 45), cv.FONT_HERSHEY_COMPLEX, 1.5, (210,247,25), 2,cv.LINE_AA)

def preprocessing_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)

    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11 , 2)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    img_resize = cv.resize(thresh, (522, 522), interpolation=cv.INTER_NEAREST)
    return img_resize


def loadModel():
    json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")

    loaded_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return loaded_model


def printSudoku(grid):
    for i in range(0,9):
        for j in range(0,9):
            print(int(grid[i][j]),end=" ")
        print()


def isSafe(grid, row, col, num):
    for i in range(0,9):
        if grid[row][i]==num:
            return False
        if grid[i][col]==num:
            return False
    secondary_row = row - row%3
    secondary_col = col - col%3
    
    for i in range(0,3):
        for j in range(0, 3):
            if(grid[i+secondary_row][j+secondary_col]==num):
                return False

    return True



def SolveSudoku(grid, row, col):
    if(row==8 and col==9):
        return True

    if (col==9):
        row += 1
        col = 0

    if(grid[row][col]>0):
        return SolveSudoku(grid, row, col+1)

    for num in range(1,10):
        if(isSafe(grid, row, col, num)):
            grid[row][col]=num

            if(SolveSudoku(grid, row, col+1)):
                return True

        grid[row][col]=0
    return False



if __name__=='__main__':   
    
    # -- loading the model --
    loaded_model = loadModel()

    # -- open camera --
    """ cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('canot open Camera') """
    

    main_img = cv.imread('sudoku_image.png')
    
    cv.imshow('main_img', main_img)

    # -- image preprocessing --

    preprocess_image = preprocessing_image(main_img)   
    cv.imshow('preprocess_image', preprocess_image)

    main_img = cv.resize(main_img, (522, 522), interpolation=cv.INTER_NEAREST)


    main_arr = np.zeros((9,9), np.uint8)
    secondary_arr = np.zeros((9,9), np.uint8)
    rows = np.vsplit(preprocess_image, 9)
    
    images = []
    for row in rows:
        col = np.hsplit(row, 9)
        images.append(col)

    images = np.array(images)
    print(images.shape)
    
    arr = np.zeros((9,9,28,28), np.float32)
        
    for k in range(0, images.shape[0]):
        for j in range(0, images.shape[1]):
            img = images[k][j][5:52,5:52]
            gray = cv.resize(img, (28,28), interpolation=cv.INTER_AREA)
            arr[k][j] = gray

    for k in range(0, 9):
        for j in range(0, 9):
            img = arr[k][j]
            X = []
            X.append(img)
            X = np.array(X)
            X = np.reshape(X, X.shape+(1,))
            predict = np.argmax(loaded_model.predict(X), axis=-1)
            main_arr[k][j] = int(predict)
            secondary_arr[k][j] = int(predict)
    
    printSudoku(main_arr)

    print()

    if(SolveSudoku(main_arr,0,0)):
        printSudoku(main_arr)
    else:
        print("No Sudoku Exist")

    display_on_board(main_arr, secondary_arr, main_img)

    main_img = cv.resize(main_img, (522, 522), interpolation=cv.INTER_NEAREST)
    cv.imshow('frame', main_img)

    cv.waitKey(0)