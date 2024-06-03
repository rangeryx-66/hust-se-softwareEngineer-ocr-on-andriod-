package com.example.myapplication;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import java.util.ArrayList;
import java.util.List;
import com.example.myapplication.ImagePreProcessor;

public class preProcessor {
    private int canvasSize;
    private double magRatio;
    private ImagePreProcessor imageProcessor;

    public preProcessor(int canvasSize, double magRatio, ImagePreProcessor imageProcessor) {
        this.canvasSize = canvasSize;
        this.magRatio = magRatio;
        this.imageProcessor = imageProcessor;
    }

    public List<Mat> preprocessImages(List<Mat> images) {
        List<Mat> imgResizedList = new ArrayList<>();
        for (Mat img : images) {
            Mat imgResized = imageProcessor.resizeAspectRatio(img, canvasSize, Imgproc.INTER_LINEAR, magRatio);
            imgResizedList.add(imgResized);
        }


        List<Mat> x = new ArrayList<>();
        for (Mat img : imgResizedList) {
            Mat normalizedImg = imageProcessor.normalizeMeanVariance(img,new Scalar(0.485, 0.456, 0.406), new Scalar(0.229, 0.224, 0.225));
            x.add(normalizedImg);
        }

        return x;
    }
    public float[] preprocessImages(Mat image) {
        Mat imgResized = imageProcessor.resizeAspectRatio(image, canvasSize, Imgproc.INTER_LINEAR, magRatio);
        Mat mat = imageProcessor.normalizeMeanVariance(image,new Scalar(0.485, 0.456, 0.406), new Scalar(0.229, 0.224, 0.225));
        int rows = mat.rows();
        int cols = mat.cols();

        float[] floatValues = new float[rows * cols * mat.channels()]; // 假设是多通道图像，每个像素可能有多个通道

        mat.convertTo(mat, CvType.CV_32FC(mat.channels())); // 将 Mat 转换为单精度浮点型

        mat.get(0, 0, floatValues); // 将 Mat 的数据复制到 float 数组中

        for (int i = 0; i < floatValues.length; ++i) {
            floatValues[i] /= 255.0f;
        }
        return floatValues;
    }
}