package com.example.myapplication;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Recognizer {

    static {
        // Load OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static List<String[]> recognizerPredict(
            OrtSession session,
            CTCLabelConverter converter,
            List<Mat> testImages,
            int batchMaxLength


    ) throws OrtException {
        List<String[]> result = new ArrayList<>();

        for (Mat image : testImages) {
            Mat imageTensor = preprocessImage(image);
            int batchSize = 1;

            long[] shape = new long[]{batchSize, 1, imageTensor.height(), imageTensor.width()};
            OnnxTensor imageOnnxTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(toFloatArray(imageTensor)), shape);

            long[] lengthForPred = new long[]{batchMaxLength};
            OnnxTensor lengthForPredTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), lengthForPred);

            long[] textForPred = new long[batchMaxLength + 1];
            OnnxTensor textForPredTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), textForPred);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("image", imageOnnxTensor);
            inputs.put("text", textForPredTensor);
            inputs.put("length", lengthForPredTensor);
            OrtSession.Result preds = session.run(inputs);
            float[][][] predsArray = (float[][][]) preds.get(0).getValue();

            // Apply softmax and filter ignore_idx
            float[][][] predsProb = softmax(predsArray);


            predsProb = normalize(predsProb);


            int[][] predsIndex = argmax(predsProb);
            int[] predsSize = new int[]{predsIndex[0].length};
            String[] predsStr = converter.decodeGreedy(flatten(predsIndex), predsSize).toArray(new String[0]);

            for (String pred : predsStr) {
                float[] maxProbs = maxProb(predsProb);
                float confidenceScore = mean(maxProbs);
                result.add(new String[]{pred, String.valueOf(confidenceScore)});
            }

        }

        return result;
    }

    private static Mat preprocessImage(Mat image) {
        // Resize and preprocess image if needed
        Imgproc.resize(image, image, new Size(100, 32)); // Example size, adjust based on your model
        Mat processedImage = new Mat();
        image.convertTo(processedImage, CvType.CV_32F);
        return processedImage;
    }

    private static float[] toFloatArray(Mat image) {
        float[] floatArray = new float[(int) (image.total() * image.channels())];
        image.get(0, 0, floatArray);
        return floatArray;
    }

    private static float[][][] softmax(float[][][] x) {
        float[][][] result = new float[x.length][x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                float max = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > max) {
                        max = x[i][j][k];
                    }
                }
                float sum = 0;
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] = (float) Math.exp(x[i][j][k] - max);
                    sum += result[i][j][k];
                }
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] /= sum;
                }
            }
        }
        return result;
    }

    private static float[][][] normalize(float[][][] x) {
        float[][][] result = new float[x.length][x[0].length][x[0][0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                float sum = 0;
                for (int k = 0; k < x[i][j].length; k++) {
                    sum += x[i][j][k];
                }
                for (int k = 0; k < x[i][j].length; k++) {
                    result[i][j][k] = x[i][j][k] / sum;
                }
            }
        }
        return result;
    }

    private static int[][] argmax(float[][][] x) {
        int[][] result = new int[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                int maxIndex = -1;
                float maxValue = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > maxValue) {
                        maxValue = x[i][j][k];
                        maxIndex = k;
                    }
                }
                result[i][j] = maxIndex;
            }
        }
        return result;
    }

    private static int[] flatten(int[][] x) {
        int[] result = new int[x.length * x[0].length];
        int index = 0;
        for (int[] row : x) {
            for (int value : row) {
                result[index++] = value;
            }
        }
        return result;
    }

    private static float[] maxProb(float[][][] x) {
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < x[i].length; j++) {
                for (int k = 0; k < x[i][j].length; k++) {
                    if (x[i][j][k] > max) {
                        max = x[i][j][k];
                    }
                }
            }
            result[i] = max;
        }
        return result;
    }

    private static float mean(float[] x) {
        float sum = 0;
        for (float value : x) {
            sum += value;
        }
        return sum / x.length;
    }

    public static void main(String[] args) throws OrtException {
        // Load model and initialize ONNX runtime
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("model.onnx", options);

        // Example usage
        List<Mat> testImages = new ArrayList<>();
        testImages.add(Imgcodecs.imread("test_image.png"));
        Map<String, String> dictList = new HashMap<>();
        dictList.put("ch_sim", "ch-pin-syl.txt");
        CTCLabelConverter converter = new CTCLabelConverter("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",new HashMap<>(),dictList);

        List<String[]> results = recognizerPredict(session, converter, testImages, 25);
        for (String[] result : results) {
            System.out.println("Prediction: " + result[0] + ", Confidence: " + result[1]);
        }
    }
}


