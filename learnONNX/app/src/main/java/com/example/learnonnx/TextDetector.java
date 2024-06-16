package com.example.learnonnx;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.*;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class TextDetector {
    public final static String TAG = "TextDetector";

    public static float[][][][] testNet(Mat[] images, OrtEnvironment ortEnv, OrtSession ortSession) throws OrtException {
        Mat image =new Mat();
        images[0].copyTo(image);
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);
        Imgproc.resize(image, image, new Size(800, 608));
        image.convertTo(image, CvType.CV_32F, 1.0 / 255.0);
        FloatBuffer buffer = FloatBuffer.allocate(3 * 800 * 608);
        image.get(0, 0, buffer.array());
        float[] imgData = new float[3 * 800 * 608];
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 608; h++) {
                for (int w = 0; w < 800; w++) {
                    imgData[c * 608 * 800 + h * 800 + w] = buffer.get((h * 800 + w) * 3 + c);
                }
            }
        }
        float[][][][] scoreTextLink = new float[0][][][];
        float[][][][] features = new float[0][][][];
        try {
            long[] shape = new long[]{1, 3, 608, 800};
            OnnxTensor tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(imgData), shape);
            Map<String, OnnxTensor> inputs = Collections.singletonMap("input", tensor);
            OrtSession.Result results = ortSession.run(inputs);
            scoreTextLink = (float[][][][]) results.get(0).getValue();
            features = (float[][][][]) results.get(1).getValue();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return scoreTextLink;
    }
}