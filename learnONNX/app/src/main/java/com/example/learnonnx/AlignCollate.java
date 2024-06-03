package com.example.learnonnx;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;


import java.util.ArrayList;
import java.util.List;

public class AlignCollate {

    private int imgH;
    private int imgW;
    private boolean keepRatioWithPad;
    private double adjustContrast;

    public AlignCollate(int imgH, int imgW, boolean keepRatioWithPad, double adjustContrast) {
        this.imgH = imgH;
        this.imgW = imgW;
        this.keepRatioWithPad = keepRatioWithPad;
        this.adjustContrast = adjustContrast;
    }

    public Mat process(Mat image) {
        if (adjustContrast > 0) {
            image = adjustContrastGrey(image, adjustContrast);
        }

        int w = image.cols();
        int h = image.rows();
        float ratio = w / (float) h;

        int resizedW;
        if (Math.ceil(imgH * ratio) > imgW) {
            resizedW = imgW;
        } else {
            resizedW = (int) Math.ceil(imgH * ratio);
        }

        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, new Size(resizedW, imgH), 0, 0, Imgproc.INTER_CUBIC);

        if (keepRatioWithPad) {
            Mat paddedImage = new Mat(imgH, imgW, CvType.CV_8UC1, Scalar.all(0));
            int padW = (imgW - resizedW) / 2;
            resizedImage.copyTo(paddedImage.colRange(padW, padW + resizedW));
            return paddedImage;
        } else {
            return resizedImage;
        }
    }

    public List<Mat> processBatch(List<Mat> batch) {
        List<Mat> processedBatch = new ArrayList<>();
        for (Mat img : batch) {
            if (img != null) {
                processedBatch.add(process(img));
            }
        }
        return processedBatch;
    }

    private Mat adjustContrastGrey(Mat image, double target) {
        Mat result = new Mat();
        image.convertTo(result, -1, target, 0);
        return result;
    }
}