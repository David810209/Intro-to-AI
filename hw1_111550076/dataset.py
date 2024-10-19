import os
import cv2

def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    for category in ['car','non-car']:
        subfolder_path = os.path.join(data_path,category)
        classfication = 1 if category == 'car' else 0
        for filename in os.listdir(subfolder_path):
          image_path = os.path.join(subfolder_path, filename)
          img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
          if img is None:
            print(f"Warn")
            continue  
          img_resized = cv2.resize(img,(36,16))
          dataset.append((img_resized,classfication))
    # End your code (Part 1)
    return dataset
