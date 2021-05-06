import sys
import fitz
import os
import cv2
import dlib

# run: & "C:/Program Files/Python37/python.exe" "c:/Users/mertd/OneDrive/Masaüstü/VS Code/Hacettepe AI Club ARGE/face_reading.py" "resume_en.pdf"

def extract_images(file_path):
    file_name = file_path.split("\\")[-1]
    file_ext = file_name.split('.')[-1]
    if file_ext == 'pdf':
        file = fitz.open(file_path)
        for page_idx in range(len(file)):
            page = file[page_idx]
            images = page.getImageList()
            if not images:
                continue
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_img = file.extractImage(xref)
                image_bytes = base_img['image']
                image_extension = base_img['ext']
                with open(os.path.join('image-%d-%d.%s' % (page_idx, img_idx, image_extension)), 'wb') as f:
                    f.write(image_bytes)
                    f.close()
    return "image-" + str(page_idx) + "-" + str(img_idx) + "." + image_extension


def facial_landmarks(output_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("-pretrained landmark detector path-")
    img = cv2.imread(output_path)
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(image=gray, box=face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=img, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
    cv2.imshow(winname="Face", mat=img)
    cv2.waitKey(delay=0)
    return landmarks


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception(f"Command must have 1 arguments! \n \n Usage: {sys.argv[0]} (filename | * | .*) output_base_directory \n Examples: \t {sys.argv[0]} text.docx images\n \t\t {sys.argv[0]} * images ")

    if (sys.argv[1].split('.')[-1] == 'pdf'):
        if sys.argv[1] in ['*', '.*']:
            files = [file for file in os.listdir(os.getcwd()) if (file.split('.')[-1] == 'pdf')]
            for file in files:
                output_path = extract_images(file)
        else:
            output_path = extract_images(sys.argv[1])
    else:
        output_path = sys.argv[1]

    landmarks = facial_landmarks(output_path)
