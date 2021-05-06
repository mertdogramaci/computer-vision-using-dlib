import cv2
import dlib

predictor_path = "-pretrained landmark detector path-"

# Initialize the default frontal face detector
detector = dlib.get_frontal_face_detector()
# Import the downloaded model
predictor = dlib.shape_predictor(predictor_path)

# set the webcam 
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Get the bounding boxes of detected faces with `detector`
    # 1 in the second argument is for upsampling, higher values would make the input bigger
    # so model could detect more faces
    dets = detector(frame, 1)

    # print("Number of detected faces: {}".format(len(dets)))

    for k, d in enumerate(dets):
        # print("Bounding box coordinates for detection {}: Letf: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))

        # get the landmarks by passing the frame and bounding box
        shape = predictor(frame, d)
        # add bounding box to the input frame
        bb_img = cv2.rectangle(frame, (d.left(), d.top()),
                               (d.right(), d.bottom()), (0, 255, 0), 2)

        # print("Landmark 0: {}, Landmark 1: {} ...".format(shape.part(0),
        #                                           shape.part(1)))

        # pretrained landmark detector can detect 68 landmarks, we'll get them one by one and 
        # put a circle into their coordinates 
        for i in range(68):
            # since cv2.circle takes coordinates as tuples and dlib shape object is an object itself, 
            # we will get the x and y coordinates separately
            x = shape.part(i).x
            y = shape.part(i).y
            # now put the circle for i_th landmark
            landmarked_img = cv2.circle(bb_img, (x, y), 1, (0, 0, 255))
            

        cv2.imshow('Video', landmarked_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
