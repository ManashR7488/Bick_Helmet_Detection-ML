
# code for only one image

from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# media path
media_path = "media/test2.jpg"

# Read your image
img = cv2.imread(media_path)

# Perform prediction
results = model.predict(source=img, imgsz=640, conf=0.5)

# Annotate results
results[0].show()  # Display the image with annotations
annotated = results[0].plot()
cv2.imwrite("annotated_image.jpg", annotated)






# Code for pridict more than one image 


# import os
# from ultralytics import YOLO
# import cv2

# # Load your trained model
# model = YOLO("best.pt")

# # Folder containing images
# image_folder = "helmetDetectionDataset/test/images"  # Change to your folder path
# output_folder = "annotated_images"  # Desired output folder

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Supported image extensions
# img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# # Loop through all images in the folder
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith(img_exts):
#         img_path = os.path.join(image_folder, filename)
#         img = cv2.imread(img_path)
#         results = model.predict(source=img, imgsz=640, conf=0.5)
#         annotated = results[0].plot()
#         # Save with a unique name (e.g., prefix "annotated_")
#         output_path = os.path.join(output_folder, f"annotated_{filename}")
#         cv2.imwrite(output_path, annotated)
#         print(f"Saved: {output_path}")



