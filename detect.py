import cv2 as cv

def detect(frame):
    haar_face = cv.CascadeClassifier(r'haar_xml\haarcascade_frontalface_default.xml')
    haar_eye_left = cv.CascadeClassifier(r'haar_xml\haarcascade_lefteye_2splits.xml')
    haar_eye_right = cv.CascadeClassifier(r'haar_xml\haarcascade_righteye_2splits.xml')

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_rect = haar_face.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 3)

    if len(face_rect):
        x, y, w, h = face_rect[0]

        face_roi = frame[y:y + h, x:x + w]

        face_roi_gray = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)

        eye_left_rect = haar_eye_left.detectMultiScale(face_roi_gray, scaleFactor= 1.6, minNeighbors= 2)
        eye_right_rect = haar_eye_right.detectMultiScale(face_roi_gray, scaleFactor= 1.6, minNeighbors= 2)

        if len(eye_left_rect) and len(eye_right_rect):
            x, y, w, h = eye_left_rect[0]
            eye_left = face_roi[y:y + h, x:x + w]
            cv.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0 , 255), 2)

            x, y, w, h = eye_right_rect[0]
            eye_right = face_roi[y:y + h, x:x + w]
            cv.rectangle(face_roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

            return 1, eye_left, eye_right, face_roi
        # else:
        #     return 0, None, None, None
            

    return 0, None, None, None

if __name__ == '__main__':

    vid = cv.VideoCapture(0)

    while True:
        _, frame = vid.read()
        cv.imshow('frame', frame)

        ret, eye_left, eye_right, face = detect(frame)

        if ret:
            cv.imshow('eye left', eye_left)
            cv.imshow('eye right', eye_right)
            #cv.imwrite('eye_open_left.jpg', eye_left)
            cv.imshow('face', face)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    cv.destroyAllWindows()
