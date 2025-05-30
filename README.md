# Visuospatial_CNN
### Line Cancellation Detection

This project detects crossed lines in a neuropsychological line cancellation test using object detection models. It assists clinicians by automating the scoring process based on participant performance.

Participants are asked to cross out lines on a standardized template. This project detects which lines are crossed. 

Key goals:
- Automatically detect and count crossed lines
- Minimize false positives (uncrossed or background elements)

#### Dataset Structure
```text
datasets/
│
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images (no labels needed)
│
├── labels/
│   ├── train/      # labels for training images
│   └── val/        # labels for validation images
│
└── data.yaml       # Dataset configuration file 
```

- All images are `.png`
- Labels are in YOLO format: `<class_id> <x_center> <y_center> <width> <height>`

- #### Preprocessing

- Images were resized and registered to the template using OpenCV.
- Annotations were created using Label Studio.


