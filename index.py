import cv2 #Pobieranie obrazu z kamery i przetwarzanie
import numpy as np #Tworzenie tablic wspomagających przetwarzanie
from scipy.ndimage.measurements import center_of_mass #Funkcja analizująca
from skimage.measure import label, regionprops #Funkcie przetwarzające

#Dodanie przeanalizowanych danych do obrazu wyjściowego
def add_text(action_frame, mass_center):
    #Zmienne potrzebne do funkcji dodającej tekst do obrazu
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerSight  = (10,200)
    fontScale              = 1
    lineType               = 2

    #Ostateczna analiza danych
    if not mass_center[0] or not mass_center[1] or not mass_center[2]:
        outText = 'wait...'
        fontColorSight = (255,255,255)
    elif (mass_center[0]+mass_center[1]+mass_center[2])/3 > 0.56:
        outText = 'Left'
        fontColorSight = (0,255,0)
    elif (mass_center[0]+mass_center[1]+mass_center[2])/3 < 0.50:
        outText = 'Right'
        fontColorSight = (0,0,255)
    else:
        outText = 'Center'
        fontColorSight = (255,0,0)
    
    #Dodanie danych/tekstu do obrazu

    cv2.putText(img,outText, 
        bottomLeftCornerSight, 
        font, 
        fontScale,
        fontColorSight,
        lineType)

#Wstępne przetwarzanie obrazu
def preprocess(action_frame):
    
    #Konwersja obrazu z przestrzeni barw BGR do HSV
    hsv = cv2.cvtColor(action_frame, cv2.COLOR_BGR2HSV) 

    #Tworzenie masek potrzebnych do przetwarzania
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    #Detekcja skóry - wyznaczanie granic
    lower_color = np.array([0, 19, 110])
    upper_color = np.array([20, 255, 255])
    
    #Faktyczna detekcja skóry
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    #Wykonanie negatywu obrazu
    mask2 = np.full_like(mask, 255) - mask
    
    #Usuwanie artefaktów - erozja
    hsv_d = cv2.erode(mask2, kernel2)
    blur = cv2.medianBlur(hsv_d, 5)
    
    return blur

#Dalsze przetwadzanie obrazu, detekcja i wyodrębnianie oczu 
def eye_detect(action_frame, thresholded_frame):
    
    #Zmienne pomocnicze
    mass_center = 0
    ey_check = -1
    min = 1000 

    eyes = eye_cascade.detectMultiScale(action_frame, 1.1, 50) #Wykrywanie oczu
    
    #usuwanie błędnie wykrytych oczu
    while len(eyes) > 2:
        for i in range(0, len(eyes)):
            if min > eyes[i][2]:
                min = eyes[i][2]
        for i in range(0, len(eyes)):
            if eyes[i][2] == min:
                eyes = np.delete(eyes, i, 0) 
                min = 1000
                break
    
    #Przenoszenie 'oczu' na pustą tablicę
    for (ex,ey,ew,eh) in eyes: 
        #Usuwanie brwi - regionów stycznych i bliskich z górną granicą 'oka'
        for region in regionprops(label(thresholded_frame[ey:ey+eh, ex:ex+ew])): 
            minr, minc, maxr, maxc = region.bbox
            if (ey_check==-1 and minr==0)or(minr==0 and ey_check<maxr): #Znajdowanie najniższego miejsca występowania brwi z regionem - wszystkie regiony znajdujące się ponad nim będą usunięte
                ey_check = maxr
            if maxr <= ey_check: #Faktyczne sprawdzenie i usunięcie brwi
                for i in range(minr, maxr):
                    for j in range(minc, maxc):
                        thresholded_frame[ey+i][ex+j] = 0
        ey_check = -1
        mass_center += center_of_mass(thresholded_frame[ey:ey+ew, ex:ex+eh])[1] / ew #Analiza pozostałych pikseli, wykrywanie środka ciężkości
    
    return mass_center/2 #Zwracanie środka ciężkości oczu

v_name = 'video_2.mp4' #Nazwa pliku
cap = cv2.VideoCapture(v_name) #Tworzenie obiektu obrazu

#Pobieranie danych obrazu potrzebnych do zapisu
v_FPS = cap.get(cv2.CAP_PROP_FPS)
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
v_format = int(cap.get(cv2.CAP_PROP_FOURCC))
v_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Obiekt zapisu obrazu
out = cv2.VideoWriter(v_name[:-4]+'_analized'+v_name[-4:],v_format, v_FPS, (v_width, v_height), True)

#Arkusz kaskadowy wykrywający oczy
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_eye.xml')

#Zmienne pomocnicze
center_array = [0,0,0]
i = 0

ret, img = cap.read() #Pobieranie obrazu z pliku
while ret:
    
    #Przetworzenie obrazu - segmentacja i operacje morfologiczne
    hsv = preprocess(img)
    
    #Analiza obrazu - wykrycie oczu i ich środka ciężkości
    mass_center = eye_detect(img, hsv)
    center_array[i%3] = mass_center
    
    #Ostateczna analiza i dodanie danych do obrazu wyjściowego
    add_text(img, center_array)
    
    #Zapisanie obrazu
    out.write(img)

    #Komunikaty zastępujące wyświetlanie obrazu
    if not i%10:
        if not i:
            print('Analyzing...')
        else:
            print(f'Analyzed {i} out of {v_frames_total} frames')
    i+=1
    ret, img = cap.read()

#Zakończenie działania programu
print('Analysis completed')
out.release()
cap.release()