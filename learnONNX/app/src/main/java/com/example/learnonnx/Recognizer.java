package com.example.learnonnx;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class Recognizer {

    public static List<String> recognize(Mat[] imgs, List<List<Point>> horizontalList, List<List<Point>> freeList, OrtEnvironment env, OrtSession session) throws OrtException {
        Mat imgCvGrey = new Mat(imgs[0].rows(), imgs[0].cols(), imgs[0].type());
        Imgproc.cvtColor(imgs[0], imgCvGrey, Imgproc.COLOR_BGR2GRAY);
        List<String> result = new ArrayList<>();
        String ignoreChar = ""; // Compute ignore_char here

        List<Object> processedImages1 = GetImageList.getImageList(horizontalList, new ArrayList<>(), imgCvGrey, 32, true);
        List<String> result1 = Collections.singletonList(getText((List<Mat>) processedImages1.get(0), (int) processedImages1.get(1), ignoreChar, env, session));
        result.addAll(result1);

        List<Object> processedImages2 = GetImageList.getImageList(new ArrayList<>(), freeList, imgCvGrey, 32, true);
        List<String> result2 = Collections.singletonList(getText((List<Mat>) processedImages2.get(0), (int) processedImages2.get(1), ignoreChar, env, session));
        result.addAll(result2);
        return result;
    }


    private static String getText(List<Mat> imageList, int maxWidth, String ignoreChar, OrtEnvironment env, OrtSession session) throws OrtException {


        AlignCollate alignCollate = new AlignCollate(32, 100, true, 0.5);
        List<Mat> processedImages = alignCollate.processBatch(imageList);

//        Map<String, String> dictList = new HashMap<>();
//        dictList.put("ch_sim", "ch-pin-syl.txt");
//        CTCLabelConverter converter = new CTCLabelConverter("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", new HashMap<>(), dictList);
        float[][][] results = predict(processedImages,session, env);
//        List<String> result = new ArrayList<>();
//        for (String[] res : results) {
//            result.addAll(Arrays.asList(res));
//        }
        return Arrays.deepToString(results);
    }
    private static float[][][] predict(List<Mat> processedImages, OrtSession session, OrtEnvironment env) throws OrtException {
        float[][][] result = new float[processedImages.size()][24][6719];

        for (int i = 0; i < processedImages.size(); i++) {
            Mat img = processedImages.get(i);
            float[] imgData = matToFloatArray(img);
            FloatBuffer floatBuffer = FloatBuffer.wrap(imgData);
            long[] shape = new long[]{1, 1, 32, 100};

            OnnxTensor tensor = OnnxTensor.createTensor(env, floatBuffer, shape);
            OrtSession.Result res = session.run(Collections.singletonMap("input.1", tensor));
            float[][] output = ((float[][][]) res.get(0).getValue())[0];

            result[i] = output;
        }

        return result;
    }
    private static float[] matToFloatArray(Mat mat) {
        int size = (int) (mat.total() * mat.channels());
        float[] array = new float[size];
        byte[] byteArray = new byte[size];
        mat.get(0, 0, byteArray);
        for (int i = 0; i < size; i++) {
            array[i] = (byteArray[i] & 0xFF) / 255.0f;
        }
        return array;
    }
//    public static List<String[]> recognizerPredict(
//            OrtSession session,
//            OrtEnvironment env,
//            CTCLabelConverter converter,
//            List<Mat> testImages,
//            int batchMaxLength) throws OrtException {
//        List<String[]> result = new ArrayList<>();
//
//        for (Mat image : testImages) {
//            Mat imageTensor = preprocessImage(image);
//            int batchSize = 1;
//
//            long[] shape = new long[]{batchSize, 1, imageTensor.height(), imageTensor.width()};
//            OnnxTensor imageOnnxTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(toFloatArray(imageTensor)), shape);
//
//            long[] lengthForPred = new long[]{batchMaxLength};
//            OnnxTensor lengthForPredTensor = OnnxTensor.createTensor(env, lengthForPred);
//
//            long[] textForPred = new long[batchMaxLength + 1];
//            OnnxTensor textForPredTensor = OnnxTensor.createTensor(env, textForPred);
//
//            Map<String, OnnxTensor> inputs = new HashMap<>();
//            inputs.put("image", imageOnnxTensor);
//            inputs.put("text", textForPredTensor);
//            inputs.put("length", lengthForPredTensor);
//            OrtSession.Result preds = session.run(inputs);
//            float[][][] predsArray = (float[][][]) preds.get(0).getValue();
//
//            // Apply softmax and filter ignore_idx
//            float[][][] predsProb = softmax(predsArray);
//
//
//            predsProb = normalize(predsProb);
//
//
//            int[][] predsIndex = argmax(predsProb);
//            int[] predsSize = new int[]{predsIndex[0].length};
//            String[] predsStr = converter.decodeGreedy(flatten(predsIndex), predsSize).toArray(new String[0]);
//
//            for (String pred : predsStr) {
//                float[] maxProbs = maxProb(predsProb);
//                float confidenceScore = mean(maxProbs);
//                result.add(new String[]{pred, String.valueOf(confidenceScore)});
//            }
//
//        }
//
//        return result;
//    }
}