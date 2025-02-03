# Word Detection and Sentence Generation Using YOLOv8 and Stable Diffusion

This project combines **real-time object detection** and **sentence generation** by using YOLOv8 for object detection and Stable Diffusion for image generation. The system detects words in real-time from a webcam feed, identifies the most repeated word, and generates a sentence using the detected words. 

The process includes:
1. Detecting objects in real-time with YOLOv8.
2. Extracting the most repeated word from the detected objects.
3. Generating a sentence based on those detected words.
4. Using **Stable Diffusion** to generate an image related to the most repeated object.

---

## Features

- **Real-Time Object Detection**: Uses YOLOv8 to detect objects in the webcam feed and extract their names.
- **Word Frequency Analysis**: Tracks and identifies the most repeated word detected from the objects.
- **Sentence Generation**: Builds a sentence based on the most repeated object(s) detected.
- **Text-to-Image Generation**: Uses the detected most repeated word to generate an image using Stable Diffusion.
- **User Interactions**: Includes the ability to clear the detected word list or quit the application.

---

## Requirements

To run the project, you'll need to install the following dependencies:

- Python 3.8 or higher
- PyTorch (with CUDA support for GPU usage)
- YOLOv8 model (`ultralytics` library)
- Diffusers library (for Stable Diffusion)
- OpenCV (for capturing video and displaying frames)
- PIL (for image generation and manipulation)

Install the required libraries using:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install diffusers
pip install ultralytics
pip install pillow
```

---

## Description

### 1. **YOLOv8 Object Detection**

The YOLOv8 model (`best.pt` weights file) is loaded to perform **real-time object detection** using the webcam feed. For each frame captured, YOLO identifies the objects present and outputs the corresponding object names.

### 2. **Word Frequency Analysis**

The names of the detected objects are stored in a list, and the program calculates the **most frequent object** (the most repeated word) detected over time. This allows the system to track which object appears most frequently.

### 3. **Sentence Generation**

Once the most frequent object(s) are detected, the program generates a sentence based on those objects. This can be useful for generating prompts for Stable Diffusion or simply creating meaningful sentences from detected words.

### 4. **Text-to-Image Generation with Stable Diffusion**

When the user presses the **'q'** key to quit, the program uses the most repeated word to create a prompt for **Stable Diffusion**. The generated image based on that prompt is then saved as a PNG file and displayed.

### 5. **Clear Detected Objects**

If the user presses the **'r'** key, the list of detected words is cleared, and the object detection process continues from a fresh state.

### 6. **Real-Time Camera Feed**

The OpenCV library is used to capture and process video frames, overlay the detected words on the live camera feed, and visualize the object detection results.

---

## Key Functions

- **most_repeated_word(word_list)**: 
  - Takes a list of detected objects and returns the most frequently occurring word (the most repeated object).
  
- **YOLOv8 Inference**:
  - Uses the YOLOv8 model to detect objects in each video frame from the webcam.

- **Sentence Generation**:
  - Based on the most repeated detected object, the program generates a sentence.

- **Text-to-Image Generation**:
  - Generates an image from the most repeated word using Stable Diffusion.

- **Real-Time Interaction**:
  - The program handles continuous object detection, interaction with the webcam, and allows the user to quit or reset the process.

---

## Installation

1. Clone or download the repository to your local machine.

   ```bash
   git clone https://github.com/yourusername/yolo-stable-diffusion-object-detection.git
   ```

2. Ensure that you have Python 3.8 or higher installed.

3. Install the required dependencies.

   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install diffusers
   pip install ultralytics
   pip install pillow
   ```

4. Ensure that the YOLOv8 model file (`best.pt`) is available in the specified directory: `C:/Users/hp/Desktop/genrative_detection_word/`.

5. Ensure that the Stable Diffusion model is available in the specified directory: `D:/stable_diffusion_model/content/stable_diffusion_model/`.

---

## How to Use

1. Run the script:

   ```bash
   python main.py
   ```

2. The webcam will start, and the system will begin detecting objects in real-time.

3. The most frequently detected object name will be displayed on the camera feed, along with the **generated sentence** (if applicable).

4. Press **'q'** to quit the application. The system will generate an image based on the most repeated object and save it as `most_repeated.png`.

5. Press **'r'** to reset and clear the detected objects list.

---

## Example Usage

### Detecting Objects:
- As the webcam captures frames, the YOLOv8 model identifies and labels objects.

### Most Repeated Object:
- The program tracks which object (or word) appears most often and updates the displayed word.

### Sentence Generation:
- The system generates a sentence based on the most frequently detected object (for example, "The cat is sitting" or "A dog is running").

### Text-to-Image Generation:
- After pressing 'q', an image based on the most repeated word (e.g., "cat", "car", "dog") is generated and saved.

---

## License

This project is open-source and available under the **MIT License**. Feel free to modify, share, and contribute to this project.

---

## Acknowledgments

- Special thanks to **Ultralytics** for providing the YOLOv8 model and **Hugging Face** for the Stable Diffusion model.
- OpenCV for real-time video processing and object detection visualization.
