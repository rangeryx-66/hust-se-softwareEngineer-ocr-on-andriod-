package com.example.myapplication;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreProcessor {
    public Mat loadImage(String imgFile) {
        Mat img = Imgcodecs.imread(imgFile, Imgcodecs.IMREAD_COLOR);
        if (img.channels() == 4) {
            Imgproc.cvtColor(img, img, Imgproc.COLOR_BGRA2BGR);
        }
        return img;
    }
    public Mat denormalizeMeanVariance(Mat inImg, Scalar mean, Scalar variance) {
        Mat img = inImg.clone();
        Core.multiply(img, variance, img);
        Core.add(img, mean, img);
        Core.multiply(img, new Scalar(255.0), img);
        img.convertTo(img, CvType.CV_8UC3);
        Core.minMaxLoc(img); // equivalent to np.clip in Python
        return img;
    }

    public Mat resizeAspectRatio(Mat img, int squareSize, int interpolation, double magRatio) {
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
    public Mat normalizeMeanVariance(Mat inImg, Scalar mean, Scalar variance) {
        Mat img = inImg.clone();
        Core.subtract(img, new Scalar(mean.val[0] * 255.0, mean.val[1] * 255.0, mean.val[2] * 255.0), img);
        Core.divide(img, new Scalar(variance.val[0] * 255.0, variance.val[1] * 255.0, variance.val[2] * 255.0), img);
        return img;
    }



    // The resize_aspect_ratio function is not directly translatable to Java as it uses specific Python features.
    // You would need to implement this manually in Java.

    public Mat cvt2HeatmapImg(Mat img) {
        img.convertTo(img, CvType.CV_8UC1);
        Imgproc.applyColorMap(img, img, Imgproc.COLORMAP_JET);
        return img;
    }
}