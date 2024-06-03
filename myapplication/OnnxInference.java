package com.example.myapplication;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OnnxInference {
    public static List<long[]> runInference(Mat[] images) throws OrtException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load model
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession session = environment.createSession("detector_craft.onnx", new OrtSession.SessionOptions());

        // Create a map for the inputs
        Map<String, OnnxTensor> inputMap = new HashMap<>();

        // Create a four-dimensional array for the batch of images
        float[][][][] batch = new float[images.length][][][];

        for (int i = 0; i < images.length; i++) {
            Mat image = images[i];

            // Load image using OpenCV and convert to RGB format
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);  // Convert to RGB format
            Imgproc.resize(image, image, new Size(800, 608));  // Resize to the expected input size of the model

            // Normalize
            image.convertTo(image, CvType.CV_32F);
            Core.divide(image, new Scalar(255.0, 255.0, 255.0), image);

            // Adjust dimension order to CxHxW
            image = image.t();
            float[][][] array = new float[image.channels()][image.cols()][image.rows()];
            for (int c = 0; c < image.channels(); c++) {
                for (int w = 0; w < image.cols(); w++) {
                    for (int h = 0; h < image.rows(); h++) {
                        array[c][w][h] = (float) image.get(w, h)[c];
                    }
                }
            }

            // Add the image data to the batch
            batch[i] = array;
        }

// Create OnnxTensor and add to the input map
        inputMap.put("input", OnnxTensor.createTensor(environment, batch));  // Assuming input name is "input"

// Run inference
        OrtSession.Result outputs = session.run(inputMap);
        List<long[]>ans = new ArrayList<>();
        // Process output
        for (int i = 0; i < outputs.size(); i++) {
            OnnxTensor output = (OnnxTensor) outputs.get(i);
            long[] shape = output.getInfo().getShape();
            ans.add(shape);

        }
        return ans;
    }
}