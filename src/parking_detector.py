import cv2
import numpy as np
import json
import time
import os
import torch
from ultralytics import YOLO

# Конфигурация путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Поднимаемся на уровень выше src/
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Создаем директории если их нет
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Файлы конфигурации
PARKING_FILE = os.path.join(DATA_DIR, "parking_spaces.json")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")
MODEL_FILE = os.path.join(MODELS_DIR, "yolov8n.pt")

def get_device(config):
    """
    Определяет доступное устройство для YOLO модели.
    Учитывает настройки из конфигурации.
    Возвращает 'cuda' если доступно, иначе 'cpu'.
    """
    force_cpu = config.get("force_cpu", False)
    device_setting = config.get("device", "auto")
    
    # Принудительное использование CPU
    if force_cpu:
        print("🔧 Принудительное использование CPU (настройка force_cpu=true)")
        return 'cpu'
    
    # Явное указание устройства
    if device_setting in ['cuda', 'cpu']:
        if device_setting == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA запрошена, но не доступна. Переключение на CPU.")
            return 'cpu'
        print(f"🔧 Использование устройства: {device_setting.upper()} (настройка device)")
        return device_setting
    
    # Автоматическое определение
    if torch.cuda.is_available():
        try:
            # Проверяем, что CUDA действительно работает
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            return 'cuda'
        except Exception:
            print("⚠️ CUDA обнаружена, но не работает. Переключение на CPU.")
            return 'cpu'
    else:
        print("ℹ️ CUDA не доступна. Использование CPU.")
        return 'cpu'

# Флаг режима отладки
debug_mode = False
# Режим редактирования парковочных мест
edit_mode = False
# Режим удаления парковочных мест
delete_mode = False
# Режим универсального распознавания (все объекты)
universal_detection = True
# Порог занятости (процент пересечения от 0.0 до 1.0)
occupancy_threshold = 0.6
# Порог для пограничного состояния (между занято и свободно)
uncertainty_threshold = 0.3
# Время в секундах для перехода в состояние "вероятно занято"
uncertainty_time_threshold = 3.0

# Настройки для частых срабатываний
frequent_detection_threshold = 10  # Количество срабатываний
frequent_detection_window = 10.0   # Временное окно в секундах

# Список ID объектов, на которые следует реагировать (только для режима автомобилей)
tracked_objects = [2, 67]  # 2=car, 67=cell phone
# Переменные для ручного выделения парковочных мест
parking_spaces = []
current_parking_space = []
drawing = False
# Индекс редактируемого парковочного места
editing_space_index = -1

# Словарь для отслеживания времени нахождения объектов в пограничном состоянии
# Формат: {space_index: {'start_time': timestamp, 'last_area': float}}
uncertainty_tracking = {}

# Словарь для отслеживания времени свободности парковочных мест
# Формат: {space_index: {'free_start_time': timestamp, 'total_free_time': float}}
free_time_tracking = {}

# Словарь для отслеживания времени занятости парковочных мест
# Формат: {space_index: {'occupied_start_time': timestamp, 'total_occupied_time': float}}
occupied_time_tracking = {}

# Словарь для отслеживания частых срабатываний в парковочных местах
# Формат: {space_index: [timestamps]}
detection_history = {}

# Словарь для отслеживания состояния частых срабатываний
# Формат: {space_index: {'frequent_detection': bool, 'last_check': timestamp}}
frequent_detection_tracking = {}

# Функция для форматирования времени в удобном формате
def format_time(seconds):
    """
    Форматирует время в удобном формате: минуты, часы, дни
    :param seconds: время в секундах
    :return: отформатированная строка времени
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:  # Менее часа
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:  # Менее суток
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:  # Более суток
        days = int(seconds // 86400)
        remaining_hours = (seconds % 86400) / 3600
        if remaining_hours > 0:
            return f"{days}d {remaining_hours:.1f}h"
        else:
            return f"{days}d"

# Функция для сохранения парковочных мест
def save_parking_spaces(spaces, file):
    with open(file, "w") as f:
        json.dump(spaces, f)

# Функция для загрузки парковочных мест
def load_parking_spaces(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Функция для сохранения конфигурации
def save_config(config, file):
    with open(file, "w") as f:
        json.dump(config, f, indent=2)

# Функция для загрузки конфигурации
def load_config(file):
    default_config = {
        "universal_detection": True,
        "occupancy_threshold": 0.6,
        "uncertainty_threshold": 0.3,
        "uncertainty_time_threshold": 3.0,
        "frequent_detection_threshold": 10,
        "frequent_detection_window": 10.0,
        "tracked_objects": [2, 67]
    }
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        save_config(default_config, file)
        return default_config

# Функция для проверки частых срабатываний
def check_frequent_detections(space_idx, current_time, threshold, window):
    """
    Проверяет, происходят ли частые срабатывания в парковочном месте
    :param space_idx: индекс парковочного места
    :param current_time: текущее время
    :param threshold: порог количества срабатываний
    :param window: временное окно в секундах
    :return: True если есть частые срабатывания
    """
    if space_idx not in detection_history:
        detection_history[space_idx] = []
    
    # Добавляем текущее время в историю
    detection_history[space_idx].append(current_time)
    
    # Удаляем старые записи (старше временного окна)
    cutoff_time = current_time - window
    detection_history[space_idx] = [t for t in detection_history[space_idx] if t > cutoff_time]
    
    # Проверяем количество срабатываний в окне
    return len(detection_history[space_idx]) >= threshold

# Функция для проверки, находится ли точка внутри полигона
def point_in_polygon(point, polygon):
    """Проверяет, находится ли точка внутри полигона"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Функция обработки мышиных событий
def draw_parking_space(event, x, y, flags, param):
    global drawing, current_parking_space, parking_spaces, tracked_objects, debug_mode, edit_mode, delete_mode, editing_space_index

    if debug_mode:
        # Отметка объектов, на которые реагирует трекер
        if event == cv2.EVENT_LBUTTONDOWN:
            for det in param.get('detections', []):
                x1, y1, x2, y2, conf, cls = det
                if x1 <= x <= x2 and y1 <= y <= y2:
                    tracked_objects.append(int(cls))  # Добавляем класс объекта
                    print(f"Добавлен класс {int(cls)} в отслеживаемые")
            return
    elif delete_mode:
        # Режим удаления парковочных мест
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, space in enumerate(parking_spaces):
                if len(space) == 4 and point_in_polygon((x, y), space):
                    del parking_spaces[i]
                    print(f"Место {i} удалено")
                    break
    elif edit_mode:
        # Режим редактирования парковочных мест
        if event == cv2.EVENT_LBUTTONDOWN:
            if editing_space_index == -1:
                # Выбор парковочного места для редактирования
                for i, space in enumerate(parking_spaces):
                    if len(space) == 4 and point_in_polygon((x, y), space):
                        editing_space_index = i
                        current_parking_space = space.copy()
                        print(f"Редактирование места {i}")
                        break
            else:
                # Редактирование выбранного места
                if len(current_parking_space) < 4:
                    current_parking_space.append((x, y))
                if len(current_parking_space) == 4:
                    parking_spaces[editing_space_index] = current_parking_space.copy()
                    print(f"Место {editing_space_index} обновлено")
                    current_parking_space = []
                    editing_space_index = -1
    else:
        # Обычный режим разметки парковочных мест
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_parking_space) < 4:
                current_parking_space.append((x, y))
            if len(current_parking_space) == 4:
                parking_spaces.append(current_parking_space)
                print(f"Место {len(parking_spaces)-1} создано")
                current_parking_space = []


def check_parking_spaces(parking_spaces, detections, tracked_objects, threshold, uncertainty_threshold, universal_detection, uncertainty_tracking, uncertainty_time_threshold, free_time_tracking, occupied_time_tracking, frequent_detection_threshold, frequent_detection_window):
    """
    Проверяет, заняты ли парковочные места с поддержкой пограничных состояний, отслеживанием времени и частых срабатываний.
    :param parking_spaces: список парковочных мест (каждое место — это список из 4 точек).
    :param detections: детектированные объекты, список [x1, y1, x2, y2, conf, cls].
    :param tracked_objects: список классов объектов для трекера.
    :param threshold: порог занятости места (0.0-1.0).
    :param uncertainty_threshold: порог для пограничного состояния (0.0-1.0).
    :param universal_detection: использовать ли универсальное распознавание.
    :param uncertainty_tracking: словарь отслеживания пограничных состояний.
    :param uncertainty_time_threshold: время для перехода в "вероятно занято".
    :param free_time_tracking: словарь отслеживания времени свободности.
    :param occupied_time_tracking: словарь отслеживания времени занятости.
    :param frequent_detection_threshold: порог частых срабатываний.
    :param frequent_detection_window: временное окно для частых срабатываний.
    :return: список состояний мест ('occupied', 'free', 'uncertain', 'frequent_detection').
    """
    current_time = time.time()
    states = []
    
    for space_idx, space in enumerate(parking_spaces):
        # Преобразуем координаты парковочного места в полигон
        pts = np.array(space, np.int32).reshape((-1, 1, 2))
        space_area = cv2.contourArea(pts)
        max_overlap_ratio = 0.0
        best_object_class = None

        # Проверяем пересечение с каждым детектированным объектом
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            
            # Фильтрация объектов в зависимости от режима
            if not universal_detection and int(cls) not in tracked_objects:
                continue

            # Преобразуем объект в прямоугольник
            obj_rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            # Проверяем пересечение
            try:
                overlap, _ = cv2.intersectConvexConvex(pts, obj_rect)
                overlap_ratio = overlap / space_area if space_area > 0 else 0
                
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    best_object_class = int(cls)
            except:
                # Если произошла ошибка при вычислении пересечения, пропускаем
                continue

        # Определяем состояние парковочного места
        if max_overlap_ratio >= threshold:
            # Место определенно занято
            state = 'occupied'
            # Очищаем отслеживание пограничного состояния
            if space_idx in uncertainty_tracking:
                del uncertainty_tracking[space_idx]
            # Очищаем отслеживание времени свободности
            if space_idx in free_time_tracking:
                del free_time_tracking[space_idx]
            # Очищаем историю детекций при полной занятости
            if space_idx in detection_history:
                del detection_history[space_idx]
            
            # Отслеживаем время занятости
            if space_idx not in occupied_time_tracking:
                # Начинаем отслеживание времени занятости
                occupied_time_tracking[space_idx] = {
                    'occupied_start_time': current_time,
                    'total_occupied_time': 0.0
                }
            else:
                # Обновляем общее время занятости
                time_since_last_update = current_time - occupied_time_tracking[space_idx]['occupied_start_time']
                occupied_time_tracking[space_idx]['total_occupied_time'] += time_since_last_update
                occupied_time_tracking[space_idx]['occupied_start_time'] = current_time
        elif max_overlap_ratio >= uncertainty_threshold:
            # Пограничное состояние - объект частично перекрывает место
            # Проверяем частые срабатывания только в пограничном состоянии
            has_frequent_detections = check_frequent_detections(space_idx, current_time, frequent_detection_threshold, frequent_detection_window)
            
            if space_idx not in uncertainty_tracking:
                # Начинаем отслеживание пограничного состояния
                uncertainty_tracking[space_idx] = {
                    'start_time': current_time,
                    'last_area': max_overlap_ratio,
                    'object_class': best_object_class
                }
                state = 'uncertain'
            else:
                # Обновляем информацию о пограничном состоянии
                uncertainty_tracking[space_idx]['last_area'] = max_overlap_ratio
                uncertainty_tracking[space_idx]['object_class'] = best_object_class
                
                # Проверяем, не прошло ли достаточно времени
                time_in_uncertainty = current_time - uncertainty_tracking[space_idx]['start_time']
                if time_in_uncertainty >= uncertainty_time_threshold:
                    state = 'uncertain_occupied'  # Вероятно занято
                else:
                    state = 'uncertain'
            
            # Если есть частые срабатывания в пограничном состоянии, считаем место занятым
            if has_frequent_detections:
                state = 'frequent_detection'
            
            # Очищаем отслеживание времени свободности в пограничном состоянии
            if space_idx in free_time_tracking:
                del free_time_tracking[space_idx]
            # Очищаем отслеживание времени занятости в пограничном состоянии
            if space_idx in occupied_time_tracking:
                del occupied_time_tracking[space_idx]
        else:
            # Место свободно
            state = 'free'
            # Очищаем отслеживание пограничного состояния
            if space_idx in uncertainty_tracking:
                del uncertainty_tracking[space_idx]
            # Очищаем отслеживание времени занятости
            if space_idx in occupied_time_tracking:
                del occupied_time_tracking[space_idx]
            # Очищаем историю детекций при освобождении места
            if space_idx in detection_history:
                del detection_history[space_idx]
            
            # Отслеживаем время свободности
            if space_idx not in free_time_tracking:
                # Начинаем отслеживание времени свободности
                free_time_tracking[space_idx] = {
                    'free_start_time': current_time,
                    'total_free_time': 0.0
                }
            else:
                # Обновляем общее время свободности
                time_since_last_update = current_time - free_time_tracking[space_idx]['free_start_time']
                free_time_tracking[space_idx]['total_free_time'] += time_since_last_update
                free_time_tracking[space_idx]['free_start_time'] = current_time

        states.append(state)
    
    return states


# Загрузка конфигурации
config = load_config(CONFIG_FILE)
universal_detection = config.get("universal_detection", True)
occupancy_threshold = config.get("occupancy_threshold", 0.6)
uncertainty_threshold = config.get("uncertainty_threshold", 0.3)
uncertainty_time_threshold = config.get("uncertainty_time_threshold", 3.0)
frequent_detection_threshold = config.get("frequent_detection_threshold", 10)
frequent_detection_window = config.get("frequent_detection_window", 10.0)
tracked_objects = config.get("tracked_objects", [2, 67])

# Определение устройства
device = get_device(config)

# Загрузка YOLO модели (отключаем вывод в консоль)
model = YOLO(MODEL_FILE, verbose=False)
model.to(device)

# Загрузка парковочных мест
parking_spaces = load_parking_spaces(PARKING_FILE)

# Видео поток
video_path = '/home/user/park_place_detector/parking1.mp4'
# video_path = 'http://10.202.34.33/webcam/?action=stream'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка подключения к видеопотоку.")
    exit(1)

cv2.namedWindow("Parking Detection")
cv2.setMouseCallback("Parking Detection", draw_parking_space)

# Основной цикл обработки видео
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Распознавание объектов (отключаем вывод в консоль)
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []

    # Инициализация переменной для состояний парковочных мест
    space_states = []

    # Отрисовка объектов в режиме отладки
    if debug_mode:
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = f"{model.names[cls]} {conf:.2f}"
            color = (255, 255, 0) if cls in tracked_objects else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Проверка парковочных мест, если не debug_mode
    if not debug_mode:
        space_states = check_parking_spaces(parking_spaces, detections, tracked_objects, 
                                          occupancy_threshold, uncertainty_threshold, 
                                          universal_detection, uncertainty_tracking, 
                                          uncertainty_time_threshold, free_time_tracking, 
                                          occupied_time_tracking, frequent_detection_threshold, 
                                          frequent_detection_window)
        
        for i, space in enumerate(parking_spaces):
            if len(space) == 4:
                pts = np.array(space, np.int32).reshape((-1, 1, 2))
                
                # Определяем цвет в зависимости от режима и состояния
                if edit_mode and i == editing_space_index:
                    color = (255, 0, 255)  # Фиолетовый для редактируемого места
                    label = "EDITING"
                elif delete_mode:
                    color = (0, 165, 255)  # Оранжевый в режиме удаления
                    label = "DELETE"
                else:
                    # Определяем цвет и подпись на основе состояния
                    if space_states[i] == 'occupied':
                        color = (0, 0, 255)  # Красный = занято
                        # Показываем время занятости
                        if i in occupied_time_tracking:
                            current_occupied_time = time.time() - occupied_time_tracking[i]['occupied_start_time']
                            total_occupied_time = occupied_time_tracking[i]['total_occupied_time'] + current_occupied_time
                            formatted_time = format_time(total_occupied_time)
                            label = f"Occupied ({formatted_time})"
                        else:
                            label = "Occupied"
                    elif space_states[i] == 'frequent_detection':
                        color = (255, 0, 255)  # Пурпурный = частые срабатывания
                        # Показываем количество срабатываний
                        if i in detection_history:
                            detections_count = len(detection_history[i])
                            label = f"Frequent ({detections_count})"
                        else:
                            label = "Frequent Detection"
                    elif space_states[i] == 'uncertain_occupied':
                        color = (0, 100, 255)  # Оранжево-красный = вероятно занято
                        label = "Probably Occupied"
                    elif space_states[i] == 'uncertain':
                        color = (0, 255, 255)  # Желтый = неопределенное состояние
                        # Показываем время в пограничном состоянии
                        if i in uncertainty_tracking:
                            time_in_uncertainty = time.time() - uncertainty_tracking[i]['start_time']
                            label = f"Uncertain ({time_in_uncertainty:.1f}s)"
                        else:
                            label = "Uncertain"
                    else:  # 'free'
                        color = (0, 255, 0)  # Зеленый = свободно
                        # Показываем время свободности
                        if i in free_time_tracking:
                            current_free_time = time.time() - free_time_tracking[i]['free_start_time']
                            total_free_time = free_time_tracking[i]['total_free_time'] + current_free_time
                            formatted_time = format_time(total_free_time)
                            label = f"Free ({formatted_time})"
                        else:
                            label = "Free"
                
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(frame, label, (space[0][0], space[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Отрисовка текущего редактируемого парковочного места
    if edit_mode and current_parking_space:
        for i, point in enumerate(current_parking_space):
            cv2.circle(frame, point, 5, (255, 0, 255), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if len(current_parking_space) > 1:
            pts = np.array(current_parking_space, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 255), thickness=2)

    # Отрисовка текущего создаваемого парковочного места
    if not debug_mode and not edit_mode and not delete_mode and current_parking_space:
        for i, point in enumerate(current_parking_space):
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if len(current_parking_space) > 1:
            pts = np.array(current_parking_space, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

    # Добавляем информационную панель
    info_text = []
    if debug_mode:
        info_text.append("DEBUG MODE - Click on objects to track")
    elif edit_mode:
        info_text.append("EDIT MODE - Click on parking space to edit")
        if editing_space_index != -1:
            info_text.append(f"Editing space {editing_space_index}")
    elif delete_mode:
        info_text.append("DELETE MODE - Click on parking space to delete")
    else:
        info_text.append("NORMAL MODE - Click to create parking spaces")
    
    # Информация о режиме детекции
    detection_mode = "Universal (All Objects)" if universal_detection else "Cars Only"
    info_text.append(f"Detection: {detection_mode}")
    info_text.append(f"Device: {device.upper()}")
    info_text.append(f"Occupancy threshold: {occupancy_threshold*100:.0f}%")
    info_text.append(f"Uncertainty threshold: {uncertainty_threshold*100:.0f}%")
    
    # Статистика парковочных мест
    if not debug_mode and space_states:
        occupied_count = sum(1 for state in space_states if state == 'occupied')
        uncertain_count = sum(1 for state in space_states if state in ['uncertain', 'uncertain_occupied'])
        frequent_count = sum(1 for state in space_states if state == 'frequent_detection')
        free_count = sum(1 for state in space_states if state == 'free')
        info_text.append(f"Spaces: {occupied_count} occupied, {uncertain_count} uncertain, {frequent_count} frequent, {free_count} free")
        
        # Статистика времени свободности и занятости
        current_time = time.time()
        free_times = []
        occupied_times = []
        
        # Собираем статистику времени свободности
        if free_time_tracking:
            for space_idx, tracking_data in free_time_tracking.items():
                if space_idx < len(space_states) and space_states[space_idx] == 'free':
                    current_free_time = current_time - tracking_data['free_start_time']
                    total_free_time = tracking_data['total_free_time'] + current_free_time
                    free_times.append(total_free_time)
        
        # Собираем статистику времени занятости
        if occupied_time_tracking:
            for space_idx, tracking_data in occupied_time_tracking.items():
                if space_idx < len(space_states) and space_states[space_idx] == 'occupied':
                    current_occupied_time = current_time - tracking_data['occupied_start_time']
                    total_occupied_time = tracking_data['total_occupied_time'] + current_occupied_time
                    occupied_times.append(total_occupied_time)
        
        # Отображаем статистику
        if free_times:
            avg_free_time = sum(free_times) / len(free_times)
            max_free_time = max(free_times)
            avg_formatted = format_time(avg_free_time)
            max_formatted = format_time(max_free_time)
            info_text.append(f"Free: avg {avg_formatted}, max {max_formatted}")
        
        if occupied_times:
            avg_occupied_time = sum(occupied_times) / len(occupied_times)
            max_occupied_time = max(occupied_times)
            avg_formatted = format_time(avg_occupied_time)
            max_formatted = format_time(max_occupied_time)
            info_text.append(f"Occupied: avg {avg_formatted}, max {max_formatted}")
    else:
        info_text.append(f"Total spaces: {len(parking_spaces)}")
    
    info_text.append("Keys: D=Debug, E=Edit, R=Delete, U=Universal, S=Save, C=Clear, Q=Quit")
    
    # Отрисовка информационной панели
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Parking Detection", frame)
    cv2.setMouseCallback("Parking Detection", draw_parking_space, param={'detections': detections})

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Выход из программы...")
        break
    elif key == ord("d"):
        debug_mode = not debug_mode
        edit_mode = False
        delete_mode = False
        editing_space_index = -1
        current_parking_space = []
        print(f"Debug: {'ON' if debug_mode else 'OFF'}")
    elif key == ord("e"):
        edit_mode = not edit_mode
        debug_mode = False
        delete_mode = False
        editing_space_index = -1
        current_parking_space = []
        print(f"Edit: {'ON' if edit_mode else 'OFF'}")
    elif key == ord("r"):
        delete_mode = not delete_mode
        debug_mode = False
        edit_mode = False
        editing_space_index = -1
        current_parking_space = []
        print(f"Delete: {'ON' if delete_mode else 'OFF'}")
    elif key == ord("s"):
        save_parking_spaces(parking_spaces, PARKING_FILE)
        print("Сохранено")
    elif key == ord("u"):
        # Переключение режима детекции
        universal_detection = not universal_detection
        # Очищаем отслеживание пограничных состояний при смене режима
        uncertainty_tracking.clear()
        free_time_tracking.clear()
        occupied_time_tracking.clear()
        detection_history.clear()
        frequent_detection_tracking.clear()
        print(f"Detection: {'Universal' if universal_detection else 'Cars only'}")
    elif key == ord("c"):
        # Сброс всех режимов
        debug_mode = False
        edit_mode = False
        delete_mode = False
        editing_space_index = -1
        current_parking_space = []
        uncertainty_tracking.clear()
        free_time_tracking.clear()
        occupied_time_tracking.clear()
        detection_history.clear()
        frequent_detection_tracking.clear()
        print("Сброс")

cap.release()
cv2.destroyAllWindows()
