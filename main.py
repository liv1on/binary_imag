
import cv2
import numpy as np

def load_image(image_path):
    #########################3####### Загрузка изображения с гранулами
    return cv2.imread(image_path)

def convert_to_grayscale(image):
    ########################################## Преобразование изображения в оттенки серого
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize_image(image):
    ############################################### Бинаризация изображения с использованием метода Оцу
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def find_contours(image):
    ################################################## Поиск контуров на бинаризованном изображении
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def calculate_area(contour, pixel_to_mm_ratio):
    ############################################### Вычисление площади контура в миллиметрах
    return cv2.contourArea(contour) * pixel_to_mm_ratio

def remove_noise(image, min_size):
    ################################################ Убирание шумов, отсеивание мелких объектов
    image = np.copy(image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            cv2.drawContours(image, [contour], 0, 0, -1)

    return image

def divide_image(image):
    ######################################### Разделение изображения на 4 одинаковые части
    height, width = image.shape[:2]
    half_width = width // 2
    half_height = height // 2

    quadrants = [
        image[0:half_height, 0:half_width],
        image[0:half_height, half_width:width],
        image[half_height:height, 0:half_width],
        image[half_height:height, half_width:width]
    ]

    return quadrants

def check_grain_distribution(image):
    ################################################# Проверка распределения гранул в каждой из зон
    pixel_to_mm_ratio = 1.0  ####################### Коэффициент конвертирования размеров из пикселей в миллиметры
    min_grain_size = 10  ################################ Минимальный размер гранулы в пикселях
    threshold_grain_size = 8  ############################ Пороговый размер гранулы в пикселях

    gray_image = convert_to_grayscale(image)
    binary_image = binarize_image(gray_image)
    noise_removed_image = remove_noise(binary_image, min_grain_size)
    quadrants = divide_image(noise_removed_image)

    result = []

    for i, quadrant in enumerate(quadrants):
        contours = find_contours(quadrant)

        if any(calculate_area(contour, pixel_to_mm_ratio) < threshold_grain_size for contour in contours):


            result.append(f"Проба не распределена в зоне {i + 1}")
        else:
            result.append("Проба распределена")

    return result



########################### Загрузка изображения
image_path = "333.jpg"
image = load_image(image_path)

############################### Проверка распределения гранул
distribution_result = check_grain_distribution(image)


for i, result in enumerate(distribution_result):
    print(f"Зона {i + 1}: {result}")




