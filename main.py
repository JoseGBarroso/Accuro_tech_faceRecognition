
import cv2 
import numpy as np 
import os

FACE_DETECTION_MODEL = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
TRAIN_PATH = "Images\Train"
TEST_PATH = "Images\Test"
FACE_DETECTED_IMAGES_PATH = "images_faces_detected"
FACE_RECOGNITION_MODEL_NAME = "face_recognition_model.yml"
FACE_RECOGNITION_TRAINED_MODEL_NAME = "trained_model/face_recognition_model.yml"



def get_task(): 
    print('''Selecciona qué deseas hacer:
    1. Utilizar el modelo entrenado.
    2. Entrenar un modelo y utilizarlo.''')

    option = input()
    try: 
        option = int(option)
        if option < 1 or option > 2: 
            1/0
    except: 
        print("ERROR: opción inválida.")
        exit()

    return option



def mk_directory_faces_detected(classes):
    if not os.path.exists(FACE_DETECTED_IMAGES_PATH):
        os.mkdir(FACE_DETECTED_IMAGES_PATH)
        for output in classes: 
            os.mkdir(os.path.join(FACE_DETECTED_IMAGES_PATH, output))



def get_train_data(classes):

    train_input = []
    train_output = []

    for person_name in classes: 
        train_path_person = os.path.join(TRAIN_PATH, person_name)
        output = classes.index(person_name)
        for image in os.listdir(train_path_person): 
            image_path = os.path.join(train_path_person, image)
            image_matrix = cv2.imread(image_path)
            grey_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
            equalized_grey_image = cv2.equalizeHist(grey_image)

            rectangles = FACE_DETECTION_MODEL.detectMultiScale(equalized_grey_image, scaleFactor=1.1, minNeighbors=5)

            max_rectangle = [0, 0, 0, 0, 0]
            
            for (i,j,width,height) in rectangles: 
                area_rectangle = width * height
                if area_rectangle > max_rectangle[4]: 
                    max_rectangle = [i, j, width, height, area_rectangle]
            if len(rectangles) > 0:
                cv2.rectangle(equalized_grey_image, (max_rectangle[0], max_rectangle[1]), (max_rectangle[0] + max_rectangle[2], max_rectangle[1] + max_rectangle[3]), (0,0,0), thickness = 2)
                cv2.imwrite(os.path.join(FACE_DETECTED_IMAGES_PATH, os.path.join(person_name, os.path.join("FD_" + image))), equalized_grey_image)
            
                input = equalized_grey_image[max_rectangle[1]:max_rectangle[1] + max_rectangle[3], max_rectangle[0]:max_rectangle[0] + max_rectangle[2]]
                train_input.append(input)
                train_output.append(output)

    train_input = np.array(train_input, dtype = "object") 
    train_output = np.array(train_output)

    print("INPUT Y OUTPUT DE ENTRENAMIENTO GENERADOS")
    print("CREADO EL DIRECTORIO CON LOS ROSTROS DETECTADOS")

    return train_input, train_output



def get_test_data(classes): 
    test_input = []
    test_output = []
    number_images_test = 0
    number_images_test_face_detected = 0
    number_images_per_class = [0] * len(classes)
    number_images_test_face_detected_per_class = [0] * len(classes)

    for person_name in classes: 
        test_path_person = os.path.join(TEST_PATH, person_name)
        output = classes.index(person_name)
        for image in os.listdir(test_path_person): 
            image_path = os.path.join(test_path_person, image)
            image_matrix = cv2.imread(image_path)
            grey_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
            equalized_grey_image = cv2.equalizeHist(grey_image)

            rectangles = FACE_DETECTION_MODEL.detectMultiScale(equalized_grey_image, scaleFactor=1.1, minNeighbors=1)

            if len(rectangles > 0): 
                inputs = []
                for (i,j,width,height) in rectangles: 
                    input = equalized_grey_image[j:j+height, i:i+width]
                    inputs.append(input)
                test_output.append(output)
                test_input.append(inputs)
                number_images_test_face_detected += 1
                number_images_test_face_detected_per_class[output] += 1
            
            number_images_test += 1
            number_images_per_class[output] += 1

    print("INPUT Y OUTPUT DE TEST GENERADOS")

    return test_input, test_output, number_images_test, number_images_test_face_detected, number_images_per_class, number_images_test_face_detected_per_class



def prediction_phase(test_input, test_output, num_classes): 
    successes_counter = 0
    successes_counter_per_class = [0] * num_classes

    for test_index in range(len(test_input)): 
        best_prediction = [1000, 0]
        for input in test_input[test_index]: 
            predicted_output, confidence = FACE_RECOGNIZER.predict(input)
            if confidence < best_prediction[0]: 
                best_prediction = [confidence, predicted_output]
        
        if best_prediction[1] == test_output[test_index]: 
            successes_counter += 1
            successes_counter_per_class[test_output[test_index]] += 1

    print("PREDICCIONES REALIZADAS")

    return successes_counter, successes_counter_per_class



def compute_accuracies(number_images_test, number_images_test_face_detected, successes_counter, successes_counter_per_class, number_images_per_class, number_images_test_face_detected_per_class, num_classes): 
    total_accuracy_only_faces_detected = successes_counter / number_images_test_face_detected
    total_accuracy = successes_counter / number_images_test
    accuracy_per_class_only_faces_detected = [0]*num_classes
    accuracy_per_class = [0]*num_classes

    for index in range(num_classes): 
        accuracy_per_class_only_faces_detected[index] = successes_counter_per_class[index] / number_images_test_face_detected_per_class[index]
    for index in range(num_classes): 
        accuracy_per_class[index] = successes_counter_per_class[index] / number_images_per_class[index]

    print("ACCURACIES CALCULADOS")

    return total_accuracy, total_accuracy_only_faces_detected, accuracy_per_class, accuracy_per_class_only_faces_detected



def print_results(number_images_test, number_images_test_face_detected, total_accuracy, total_accuracy_only_faces_detected, accuracy_per_class, accuracy_per_class_only_faces_detected, classes): 
    print()
    print(f"Porcentaje de imágenes en los que se ha detectado al menos una cara supuestamente: {number_images_test_face_detected * 100/number_images_test}%")
    print(f"Accuracy total: {total_accuracy * 100}%")
    print(f"Accuracy total descartando las imágenes en las que no se ha detectado ningún rostro: {total_accuracy_only_faces_detected * 100}%")
    print("Accuracy por clase:")
    for index in range(len(classes)): 
        print(f"    Accuracy en imágenes de {classes[index]}: {accuracy_per_class[index] * 100}% y {accuracy_per_class_only_faces_detected[index] * 100}% descartando las imágenes en las que no se ha detectado ningún rostro.")



def main(): 
    option = get_task()

    classes = ["AndySamberg", "BillieEilish", "BradPitt", "CourtneyCox", "DwayneJohnson", "ElizabethOlsen", "HenryCavill", "HughJackman", "LisaKudrow", "NataliePortman"]

    if option == 1: 
        FACE_RECOGNIZER.read(FACE_RECOGNITION_TRAINED_MODEL_NAME)
        print("LECTURA DEL MODELO REALIZADA")
    else: 
        mk_directory_faces_detected(classes)
        train_input, train_output = get_train_data(classes)
        FACE_RECOGNIZER.train(train_input, train_output)
        print("ENTRENAMIENTO FINALIZADO")
        FACE_RECOGNIZER.save(FACE_RECOGNITION_MODEL_NAME)
        print("MODELO DE RECONOCIMIENTO DE ROSTROS GUARDADO")

    test_input, test_output, number_images_test, number_images_test_face_detected, number_images_per_class, number_images_test_face_detected_per_class = get_test_data(classes)
    successes_counter, successes_counter_per_class = prediction_phase(test_input, test_output, len(classes))
    total_accuracy, total_accuracy_only_faces_detected, accuracy_per_class, accuracy_per_class_only_faces_detected = compute_accuracies(number_images_test, number_images_test_face_detected, successes_counter, successes_counter_per_class, number_images_per_class, number_images_test_face_detected_per_class, len(classes))
    print_results(number_images_test, number_images_test_face_detected, total_accuracy, total_accuracy_only_faces_detected, accuracy_per_class, accuracy_per_class_only_faces_detected, classes)



main()




