import cv2
import mediapipe as mp
import time


# Инициализация Mediapipe для отслеживания руки
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

keyboard_keys = [str(i) for i in range(1, 10)] + ["0"]  # Создаем клавиатуру
key_size = 45

buffer = '' # Записанные цифры
last_press_time = time.time()  # Время последнего нажатия

frame = cv2.VideoCapture(0)

while True:
    state, image = frame.read()
    h, w, _ = image.shape
    image_copy = image.copy()
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Расчет отступа для центрирования клавиатуры по горизонтали
    keyboard_width = 3 * key_size + 2 * 10
    offset_x = (w - keyboard_width) // 2

    # Позиции клавиш с учетом центрирования
    key_positions = [(offset_x + (i % 3) * (key_size + 20), 200 + (i // 3) * (key_size + 20)) for i in range(9)]
    key_positions.append((offset_x + key_size + 20, 200 + 3 * (key_size + 20)))  # Позиция "0" под "8"

    
     # Распознавание руки и получение координат кончика указательного пальца
    results = hands.process(image_copy)
    index_finger_tip = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(image_copy, (index_x, index_y), 5, (255, 0, 0), -1)  # Маркер для кончика пальца



    # Отображение клавиатуры на экране
    for i in range(len(keyboard_keys)):
        key = keyboard_keys[i]
        x, y = key_positions[i]
        color = (60, 100, 100)  # Цвет клавиш

        # Проверка, находится ли указательный палец на клавише
        if index_finger_tip:
            if x <= index_x <= x + key_size and y <= index_y <= y + key_size:
                color = (60, 255, 255)  # Подсвечивание клавиши
                # Проверка задержки для фиксации нажатия
                if time.time() - last_press_time > 3:
                    buffer += key
                    last_press_time = time.time()  # Обновление времени последнего нажатия

        # Отображение клавиши на экране
        cv2.rectangle(image_copy, (x, y), (x + key_size, y + key_size), color, -1)
        cv2.putText(image_copy, key, (x + 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Вывод буфера с введенными цифрами
    cv2.putText(image_copy, f"Password: {buffer}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        

     # Отображение результата
    cv2.imshow("Virtualnaya Klaviatura", image_copy)

    if cv2.waitKey(25) == 27:
            break

frame.release()
cv2.destroyAllWindows()