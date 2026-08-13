import cv2
import os

image_path = os.path.join("assets", "patient.jpg")

print("Current Working Directory:", os.getcwd())
print("Image Path:", image_path)

image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found!")
else:
    print("✅ Image loaded successfully!")
    print("Image Shape:", image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(image, (400, 400)) 
    cv2.rectangle(image, (40, 40), (160, 160), (0, 255, 0), 2) 
    cv2.putText(
    image,
    "Patient",
    (40, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
) 

    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Resized Image", resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()