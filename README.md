# HUST-SE-SoftwareEngineering-OCR-on-Android-2024

### Members: 杨轩, 万昌麟

## Goals of the Project

- [ ] **Task 1: Download and Understand OCR Model**
  - 📥 [OCR Model Download and Implementation](https://github.com/hlp-ai/yimt/tree/main/ocr)
- [ ] **Task 2: Optimize the Model**
  - ✂️ Trim the model for lightweight implementation
- [ ] **Task 3: Familiarize with Android Development**
  - 📱 [Android Studio Installation and Setup](https://developer.android.google.cn/studio/install)
- [ ] **Task 4: Implement Android Frontend Design and Code**
  - 🎨 Design and code the frontend for Android
- [ ] **Task 5: Android Studio Implementation**
  - 🛠️ Use Android Studio to preprocess images, run CRAFT inference, extract cropped character images, perform CRNN inference, decode, and output to frontend
- [ ] **Task 6: Export Android APK File**
  - 📦 Export the final APK file
## Code Architecture

The code structure under `learnONNX/app/src/main/java/com/example/learnonnx` is as follows:

- **MainActivity.java**: The main function, responsible for:
  - Initializing the UI
  - Designing interface interactions
  - Requesting image reading and photographing from the Android system
  - Invoking OCR functionality
  - Updating images with bounding boxes
  - Outputting results to a text box
- **ProcessOCR.java**: Executes OCR by calling `detector`, `postProcess`, and `recognizer`, handling parameter and result transmission.
- **TextDetector.java**: Executes the detector, including image preprocessing and inference using the ONNX model.
- **PostProcess.java**: Processes the results from the CRAFT model to generate coordinates for detected sub-images containing characters.
- **GetImageList.java**: Crops and transforms images based on detected sub-image coordinates to fit the CRNN model input format.
- **AlignCollate.java**: Preprocesses cropped images, either transforming them if too long or padding the right side to fit the CRNN input.
- **Recognizer.java**: Calls `GetImageList` and `AlignCollate`, and infers the CRNN results.
- **TextDecoder.java**: Decodes the CRNN results to get the final output.
## Project Structure

```plaintext
hust-se-softwareEngineering-ocr-on-android-2024/
│
├── model/                   # Directory for the OCR model
├── android/                 # Android project directory
│   ├── app/                 # Main app directory
│   ├── build/               # Build output
│   └── ...                  # Other directories and files
├── README.md                # Project README file
└── ...                      # Other files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
