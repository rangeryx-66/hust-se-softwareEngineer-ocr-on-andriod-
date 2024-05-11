package com.example.learnonnx;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.*;
import java.util.ArrayList;
import java.util.*;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class TextDetector{
    public static float[][][][] testNet(int canvasSize, float magRatio,  Mat[] images,byte[] model) {
        List<Mat> images_resized = new ArrayList<>();
        for (Mat image:images) {
            images_resized.add( resizeAspectRatio(image, canvasSize, Imgproc.INTER_LINEAR, magRatio));
        }
        List<Mat> processedImages = new ArrayList<>();
        for (Mat n_img : images_resized) {
            Mat normalizedImg = normalizeMeanVariance(n_img,new Scalar(0.485, 0.456, 0.406), new Scalar(0.229, 0.224, 0.225));
            Mat transposedImg = transposeAndPermuteDims(normalizedImg, new int[]{2, 0, 1});
            processedImages.add(transposedImg);
        }
        List<Mat> tensors = new ArrayList<>();
        float[][][][]ans=new float[1][][][];
        // 将每个矩阵转换为 PyTorch 张量
        for (Mat mat : processedImages) {
            Mat tensor = convertToTensor(mat);
            tensors.add(tensor);
        }
        try {
            // 1. 加载模型
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            OrtSession session = env.createSession(model, options);

            // 2. 准备输入
            float[][] inputs = new float[images.length][];
            for (int i = 0; i < tensors.size(); i++) {
                Mat tensor = tensors.get(i);
                float[] inputData = new float[tensor.rows() * tensor.cols()];
                tensor.get(0, 0, inputData);
                inputs[i] = inputData; // Assuming 1 input channel
            }
            // 创建输入容器
            Map<String, OnnxTensor> container = new HashMap<>();
            for (int i = 0; i < images.length; i++) {
                OnnxTensor tensor = OnnxTensor.createTensor(env, inputs[i]);
                List<String> inputNamesList = new ArrayList<>(session.getInputNames());
                container.put(inputNamesList.get(i), tensor);
            }

            // 执行推理
            OrtSession.Result output = session.run(container);
            // 获取输出
            OnnxTensor outputTensor = (OnnxTensor) output.get(0);
            float[][][][] outputData = (float[][][][]) outputTensor.getValue();
            // 5. 关闭会话和环境
            session.close();
            env.close();
            ans=outputData;
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return ans;

    }
    private static Mat convertToTensor(Mat mat) {
        // 假设 mat 是单通道的浮点型矩阵
        Mat tensor = new MatOfFloat();
        mat.convertTo(tensor, CvType.CV_32F); // 将矩阵转换为 32 位浮点型
        return tensor;
    }
    private static Mat transposeAndPermuteDims(Mat img, int[] permutation) {
        List<Mat> channels = new ArrayList<>();

        // 拆分通道
        Core.split(img, channels);

        // 重新排列通道
        Mat transposed = new Mat();
        Core.merge(new ArrayList<>(Arrays.asList(channels.get(permutation[0]), channels.get(permutation[1]), channels.get(permutation[2]))), transposed);

        return transposed;
    }
    public static Mat resizeAspectRatio(Mat img, int squareSize, int interpolation, double magRatio) {
        int height = img.rows();
        int width = img.cols();
        int channel = img.channels();

        // magnify image size
        double targetSize = magRatio * Math.max(height, width);

        // set original image size
        if (targetSize > squareSize) {
            targetSize = squareSize;
        }

        double ratio = targetSize / Math.max(height, width);

        int targetH = (int)(height * ratio);
        int targetW = (int)(width * ratio);
        Mat resizedImg = new Mat();
        Size targetSize1 = new Size(targetW, targetH);
        Imgproc.resize(img, resizedImg, targetSize1, 0, 0, interpolation);

        // make canvas and paste image
        int targetH32 = targetH % 32 == 0 ? targetH : targetH + (32 - targetH % 32);
        int targetW32 = targetW % 32 == 0 ? targetW : targetW + (32 - targetW % 32);
        Mat canvas = Mat.zeros(targetH32, targetW32, img.type());
        resizedImg.copyTo(new Mat(canvas, new Rect(0, 0, targetW, targetH)));

        Size sizeHeatmap = new Size(targetW / 2, targetH / 2);

        return canvas;
    }

    public static Mat normalizeMeanVariance(Mat inImg, Scalar mean, Scalar variance) {
        Mat img = inImg.clone();
        Core.subtract(img, new Scalar(mean.val[0] * 255.0, mean.val[1] * 255.0, mean.val[2] * 255.0), img);
        Core.divide(img, new Scalar(variance.val[0] * 255.0, variance.val[1] * 255.0, variance.val[2] * 255.0), img);
        return img;
    }

    private static List<Rect> getDetBoxes(Mat scoreText, Mat scoreLink, float textThreshold, float linkThreshold,
                                          boolean lowText, boolean poly, boolean estimateNumChars,
                                          float ratioW, float ratioH) {
        // Implement post-processing here
        return new ArrayList<>();
    }
}