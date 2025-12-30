import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# Создаем объект класса для нахождение точек на лице
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


frame = cv2.VideoCapture(0)



drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    # for idx, file in enumerate(image_files):
    while True:
        state, image = frame.read()
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Накладываем точки на лицо
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
        # results.multi_face_landmarks - это перебираемый объект, поэтому мы не
        # сможем обратиться к отдельной точке по индексу
        # Посчитаем, сколько координат вернула функция и сохраним координаты носа
            c = 0
            for i, point in enumerate(face_landmarks.landmark):
                if c == 4:
                    nose = point
                elif c == 33:
                    l_2 = point # left eye inside
                elif c == 133:
                    l_1 = point # left eye outside
                elif c == 159:
                    l_4 = point # left eye top
                elif c == 145:
                    l_3 = point # left eye bottom

                elif c == 362:
                    r_2 = point # right eye outside
                elif c == 263:
                    r_1 = point # right eye inside
                elif c == 386:
                    r_4 = point # right eye top
                elif c == 374:
                    r_3 = point # right eye bottom
                c += 1
            # print('Количество точек:', c)
            # print('Точка носа:\n', nose)

                # Рисуем точку с носом
        h, w, c = annotated_image.shape
        # cx, cy = int(nose.x * w), int(nose.y * h)
        # cv2.circle(annotated_image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        # Вычисляем центр левого глаза
        l_eye_x = ((l_1.x * l_2.y - l_1.y * l_2.x)*(l_3.x - l_4.x) - (l_1.x - l_2.x)*(l_3.x * l_4.y - l_3.y * l_4.x)) / ((l_1.x - l_2.x)*(l_3.y - l_4.y) - (l_1.y - l_2.y)*(l_3.x - l_4.x))
        l_eye_y = ((l_1.x * l_2.y - l_1.y * l_2.x)*(l_3.y - l_4.y) - (l_1.y- l_2.y)*(l_3.x * l_4.y - l_3.y * l_4.x)) / ((l_1.x - l_2.x)*(l_3.y - l_4.y) - (l_1.y - l_2.y)*(l_3.x - l_4.x))
        
        # l_cx, l_cy = int(l_eye_x * w), int(l_eye_y * h)
        # cv2.circle(annotated_image, (l_cx, l_cy), 5, (255, 0, 0), cv2.FILLED)


        # Вычисляем центр правого глаза
        r_eye_x = ((r_1.x * r_2.y - r_1.y * r_2.x)*(r_3.x - r_4.x) - (r_1.x - r_2.x)*(r_3.x * r_4.y - r_3.y * r_4.x)) / ((r_1.x - r_2.x)*(r_3.y - r_4.y) - (r_1.y - r_2.y)*(r_3.x - r_4.x))
        r_eye_y = ((r_1.x * r_2.y - r_1.y * r_2.x)*(r_3.y - r_4.y) - (r_1.y- r_2.y)*(r_3.x * r_4.y - r_3.y * r_4.x)) / ((r_1.x - r_2.x)*(r_3.y - r_4.y) - (r_1.y - r_2.y)*(r_3.x - r_4.x))
        
        # r_cx, r_cy = int(r_eye_x * w), int(r_eye_y * h)
        # cv2.circle(annotated_image, (r_cx, r_cy), 5, (255, 0, 0), cv2.FILLED)

        l2_l1 = ((l_2.x - l_1.x)**2 + (l_2.y - l_1.y)**2)**0.5*1000 # Ширина глаза
        
        face_image = Image.fromarray(annotated_image)
        # Добавляем нос
        mask = Image.open('Red-Nose.png')
        image_nose = cv2.imread('Red-Nose.png', cv2.IMREAD_COLOR)
        np_nose_c = np.array(image_nose)
        image_nose_b = Image.fromarray(np_nose_c)


        nose_width = int(l2_l1)
        nose_height = int(l2_l1*0.7)

        image_nose_b = image_nose_b.resize((nose_width, nose_height), Image.Resampling.LANCZOS)
        mask = mask.resize((nose_width, nose_height), Image.Resampling.LANCZOS)

        # Вычисляем координаты, откуда нужно будет рисовать нос
        nx = int(nose.x * w - nose_width / 2)
        ny = int(nose.y * h - nose_height / 2)

        # Добавляем звёзды
        mask_s = Image.open('star.png')
        image_star = cv2.imread('star.png', cv2.IMREAD_COLOR)
        np_star_c = np.array(image_star)
        image_star_b = Image.fromarray(np_star_c)
        star_size = int(l2_l1 * 0.3)


        image_star_b = image_star_b.resize((star_size, star_size), Image.Resampling.LANCZOS)
        mask_s = mask_s.resize((star_size, star_size), Image.Resampling.LANCZOS)
        # Вычисляем координаты, откуда нужно будет рисовать левый глаз
        lx = int(l_eye_x * w - star_size / 2)
        ly = int(l_eye_y * h - star_size / 2)
        # Вычисляем координаты, откуда нужно будет рисовать правый глаз
        rx = int(r_eye_x * w - star_size / 2)
        ry = int(r_eye_y * h - star_size / 2)

        # Вставляем все на видео
        face_image.paste(image_star_b, (lx,ly), mask_s)
        face_image.paste(image_star_b, (rx,ry), mask_s)
        face_image.paste(image_nose_b, (nx,ny), mask)
        face_image_np = np.array(face_image)

        cv2.imshow('Face', face_image_np)

        k = cv2.waitKey(30)
        if k == 27:
            break