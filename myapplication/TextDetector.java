package com.example.myapplication;

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
    public static float[][][][] testNet(int canvasSize, float magRatio,  Mat[] images) {
        List<Mat> images_resized = new ArrayList<>();
        for (Mat image:images) {
            Imgproc.resize(image, image, new Size(800, 608));
            images_resized.add(image);
        }
        List<Mat> processedImages = new ArrayList<>();
        for (Mat n_img : images_resized) {
            //Mat normalizedImg = normalizeMeanVariance(n_img, new double[]{0.485, 0.456, 0.406}, new double[]{0.229, 0.224, 0.225});
            Mat transposedImg = transposeAndPermuteDims(n_img, new int[]{2, 0, 1});
            processedImages.add(transposedImg);
        }
        List<Mat> tensors = new ArrayList<>();

        // 将每个矩阵转换为 PyTorch 张量
        for (Mat mat : processedImages) {
            Mat tensor = convertToTensor(mat);
            tensors.add(tensor);
        }
        float[][][][] outputData = new float[images.length][][][];
        try {
            // 1. 加载模型
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            OrtSession session = env.createSession("E:\\MyApplication\\app\\src\\main\\res\\detector_craft.onnx", options);
            // 2. 准备输入
            float[][][][] inputs = new float[images.length][][][];
            for (int i = 0; i < tensors.size(); i++) {
                Mat image = tensors.get(i);
                float[][][] array = new float[image.channels()][image.cols()][image.rows()];
                for (int c = 0; c < image.channels(); c++) {
                    for (int w = 0; w < image.cols(); w++) {
                        for (int h = 0; h < image.rows(); h++) {
                            array[c][w][h] = (float) image.get(w, h)[c];
                        }
                    }
                }
                inputs[i] = array;
            }
            // 创建输入容器
            Map<String, OnnxTensor> container = new HashMap<>();
            for (int i = 0; i < images.length; i++) {
                OnnxTensor tensor = OnnxTensor.createTensor(env, inputs);
                container.put("input", tensor);
            }

            // 执行推理
            OrtSession.Result output = session.run(container);
            // 获取输出
            outputData = (float[][][][]) output.get(0).getValue();

            // 5. 关闭会话和环境
            session.close();
            env.close();

        } catch (OrtException e) {
            e.printStackTrace();
        }
        return outputData;

    }
    public static List<List<List<int[]>>>postProcess(float[][][][] outputData, Mat[] images, int canvasSize, float magRatio) {
        float[][][][] y=outputData;


        float text_threshold=0.2f, link_threshold=0.4f, low_text=0.4f;
        boolean poly=false;
        float ratio_w=1,ratio_h=1;

        List<List<MatOfPoint2f>> boxes_list = new ArrayList<>();


        for (int i = 0; i < y.length; i++) {
            float[][][] out = y[i];
            float[][] score_text1 = out[0];
            float[][] score_link1 = out[1];
            Mat score_text = new Mat(score_text1.length, score_text1[0].length, CvType.CV_32F);
            Mat score_link = new Mat(score_link1.length, score_link1[0].length, CvType.CV_32F);
            for (int j = 0; j < score_text1.length; j++) {
                for (int k = 0; k < score_text1[0].length; k++) {
                    score_text.put(j, k, score_text1[j][k]);
                    score_link.put(j, k, score_link1[j][k]);
                }
            }

            // Assuming getDetBoxes is defined and returns boxes, polys, mapper
            List<MatOfPoint2f> boxes = Postcore.getDetBoxesCore(score_text, score_link, text_threshold, link_threshold, low_text);
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h);
            boxes_list.add(boxes);


        }
        List<List<int[]>> result = new ArrayList<>();

        // 遍历 polys_list 并将每个多边形转换为整数数组并添加到 result 列表中
        for (List<MatOfPoint2f> polys : boxes_list) {
            List<int[]> single_img_result = new ArrayList<>();
            for (MatOfPoint2f box : polys) {
                // 将 MatOfPoint2f 转换为整数数组
                int[] poly1 = convertMatOfPoint2fToIntArray(box);
                single_img_result.add(poly1);
            }
            result.add(single_img_result);
        }
        List<List<int[]>> horizontalListAgg = new ArrayList<>();
        List<List<int[]>> freeListAgg = new ArrayList<>();
        List<List<List<int[]>>>ans=new ArrayList<>();
        for (List<int[]>list : result)
        {
            List<Object>hflist=TextBoxGrouper.groupTextBox(list);
            List<int[]>horizontalList=(List<int[]>)hflist.get(0);
            List<int[]>freeList=(List<int[]>)hflist.get(1);
            Iterator<int[]> iterator = horizontalList.iterator();
            while (iterator.hasNext()) {
                int[] item = iterator.next();
                if (Math.max(item[1] - item[0], item[3] - item[2]) <= 20) {
                    iterator.remove();
                }
            }

            // Filter freeList
            iterator = freeList.iterator();
            while (iterator.hasNext()) {
                int[] item = iterator.next();
                double max_x = Double.MIN_VALUE, min_x = Double.MAX_VALUE;
                double max_y = Double.MIN_VALUE, min_y = Double.MAX_VALUE;
                for (int i = 0; i < item.length; i += 2) {
                    max_x = Math.max(max_x, item[i]);
                    min_x = Math.min(min_x, item[i]);
                    max_y = Math.max(max_y, item[i + 1]);
                    min_y = Math.min(min_y, item[i + 1]);
                }
                if (Math.max(max_x - min_x, max_y - min_y) <= 20) {
                    iterator.remove();
                }
            }
            horizontalListAgg.add(horizontalList);
            freeListAgg.add(freeList);

        }
        ans.add(horizontalListAgg);
        ans.add(freeListAgg);
        return ans;
    }
    private static int[] convertMatOfPoint2fToIntArray(MatOfPoint2f box) {
        int[] poly = new int[box.rows() * box.cols()];
        float[] points = new float[box.rows() * box.cols() * 2];
        box.get(0, 0, points);

        for (int i = 0; i < points.length; i += 2) {
            poly[i] = Math.round(points[i]);
            poly[i + 1] = Math.round(points[i + 1]);
        }

        return poly;
    }
    private static List<MatOfPoint2f> adjustResultCoordinates(List<MatOfPoint2f>polys, float ratio_w, float ratio_h) {
        List<MatOfPoint2f>ans=new ArrayList<>();
        if (polys != null && polys.size() > 0) {
            for (int i = 0; i < polys.size(); i++) {
                MatOfPoint2f poly = polys.get(i);
                for (int j = 0; j < poly.rows(); j++) {
                    double[] point = poly.get(j, 0);
                    point[0] *= ratio_w;
                    point[1] *= ratio_h;
                    poly.put(j, 0, point);
                }
                ans.add(poly);
            }
        }
        return polys;
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

    public static Mat normalizeMeanVariance(Mat inImg, double[] mean, double[] variance) {
        // should be RGB order
        Mat img = inImg.clone();

        // Convert image to float32
        img.convertTo(img, CvType.CV_32F);

        // Subtract mean
        Core.subtract(img, new Scalar(mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0), img);

        // Divide by variance
        Core.divide(img, new Scalar(variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0), img);

        return img;
    }


}
