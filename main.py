import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import cv2
from ultralytics import YOLO

def most_repeated_word(word_list):
    word_count = {}
    
    # Count occurrences of each word
    for word in word_list:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # Find the word with the maximum count
    max_count = 0
    most_repeated_word = None
    for word, count in word_count.items():
        if count > max_count:
            max_count = count
            most_repeated_word = word
    
    return most_repeated_word

# Load the YOLOv8 model
model = YOLO('C:\\Users\\hp\\Desktop\\genrative_detection_word\\best.pt')
 
# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Check if GPU is available
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# Load the Stable Diffusion model
stable_diff_pipe = StableDiffusionPipeline.from_pretrained("D:\\stable_diffusion_model\\content\\stable_diffusion_model")
stable_diff_pipe.to(torch_device)

# Define the guidance scale
guidance_scale = 12.5

# Initialize a list to hold detected object names
detected_objects = []

# Loop through the camera frames
while True:
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Extract names of detected objects
        for result in results:
            for obj in result.boxes.data:
                class_id = int(obj[5])
                class_name = model.names[class_id]
                detected_objects.append(class_name)

        # Print the list of detected objects
        print(detected_objects)

        # Find the most repeated word
        most_repeated = most_repeated_word(detected_objects)
        print(f"Most repeated word: {most_repeated}")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the most repeated word on the frame
        cv2.putText(annotated_frame, f"Most Repeated: {most_repeated}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Capture key press
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if key == ord("q"):
            # Release camera capture object and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

            # Generate the image based on the most repeated word
            prompt = [most_repeated]
            with autocast("cuda"):
                image = stable_diff_pipe(prompt, guidance_scale=guidance_scale).images[0]

            # Save and display the generated image
            image.save(f"{most_repeated}.png")
            print(f"Image saved as '{most_repeated}.png'")
            image.show()
            
            break
        # Clear the list if 'r' is pressed
        elif key == ord("r"):
            detected_objects = []
            print("Detected objects list cleared.")

    else:
        print("Error: Could not read frame from camera.")
        break
