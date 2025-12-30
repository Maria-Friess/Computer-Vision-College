import cv2
import mediapipe as mp
import math


# Иницилизируем класс нахождения точек тела (поз)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
sits = 0
line = False

BG_COLOR = (192, 192, 192) # gray

video_path = 'sit48024.mp4'

frame = cv2.VideoCapture(video_path)

# Начинаем находить позы на изображении с помощью метода Pose
with mp_pose.Pose(
  static_image_mode=True,
  model_complexity=2,
  enable_segmentation=True,
  min_detection_confidence=0.5) as pose:
    while(frame.isOpened()):
      state, image = frame.read()

      # Читаем файл и получаем размеры изрбражения
      image_height, image_width, _ = image.shape
      # Преобразуем из модели BGR в RGB
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      annotated_image = image.copy()
      pose_landmarks_list = results.pose_landmarks

    
      count = 0
      for point in results.pose_landmarks.landmark:
          if count == 24:
            r_h = point
          elif count == 26:
            r_k = point
          elif count == 28:
            r_a = point
          count += 1

      h, w, c = annotated_image.shape
      # Рисуем точки для правой ноги
      r_h_x, r_h_y = int(r_h.x * w), int(r_h.y * h)
      cv2.circle(annotated_image, (r_h_x, r_h_y), 2, (255, 0, 0), cv2.FILLED)
      r_k_x, r_k_y = int(r_k.x * w), int(r_k.y * h)
      cv2.circle(annotated_image, (r_k_x, r_k_y), 2, (255, 0, 0), cv2.FILLED)
      r_a_x, r_a_y = int(r_a.x * w), int(r_a.y * h)
      cv2.circle(annotated_image, (r_a_x, r_a_y), 2, (255, 0, 0), cv2.FILLED)


      # Находим угол сгиба в колене для правой ноги
      r_l_v1_x, r_l_v1_y = r_h.x - r_k.x, r_h.y - r_k.y  # Первый вектор, образущий угол 
      r_l_v2_x, r_l_v2_y = r_a.x - r_k.x, r_a.y - r_k.y  # Второй вектор, образущий угол 
      r_l_sm = r_l_v1_x*r_l_v2_x + r_l_v1_y*r_l_v2_y  # Находим скалярное произведение векторов
      r_l_v1_len = (r_l_v1_x**2 + r_l_v1_y**2)**0.5  # Длина первого вектора
      r_l_v2_len = (r_l_v2_x**2 + r_l_v2_y**2)**0.5  # Длина второго вектора
      r_l_cos = r_l_sm / (r_l_v1_len * r_l_v2_len)  # Находим косинус угла 
      r_l_angle = math.acos(r_l_cos)
      print(r_l_angle)

      # Считаем приседы
      if r_l_angle < 1.8:
        line = True
      elif r_l_angle > 1.8 and line:
        sits += 1
        line = False


      # Выводим угол и кол-во приседаний
      cv2.putText(annotated_image, f'Angle (radians): {round(r_l_angle, 3)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      cv2.putText(annotated_image, f'Squats: {sits}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 50, 255), 2)
      
      cv2.imshow('Sits', annotated_image)
      if cv2.waitKey(25) == 27:
        break

    frame.release()
    cv2.destroyAllWindows()
print(sits)