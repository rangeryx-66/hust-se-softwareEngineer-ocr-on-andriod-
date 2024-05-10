package com.example.learnonnx;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

public class ImageProcessor {
    public Bitmap[] reformatInput(Object image) throws IOException {
        Bitmap img = null;
        Bitmap imgCvGrey = null;

        if (image instanceof String) {
            String imagePath = (String) image;
            if (imagePath.startsWith("http://") || imagePath.startsWith("https://")) {
                // Download the image and decode it
                URL url = new URL(imagePath);
                InputStream input = url.openStream();
                img = BitmapFactory.decodeStream(input);
                input.close(); // Close the stream
            } else {
                File imgFile = new File(imagePath);
                img = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
            }
            imgCvGrey = convertToGrayScale(img);
        } else if (image instanceof byte[]) {
            byte[] imageData = (byte[]) image;
            img = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);
            imgCvGrey = convertToGrayScale(img);
        } else if (image instanceof Bitmap) {
            img = (Bitmap) image;
            imgCvGrey = convertToGrayScale(img);
        } else {
            throw new IllegalArgumentException("Unsupported image format");
        }

        return new Bitmap[]{img, imgCvGrey};
    }

    private Bitmap convertToGrayScale(Bitmap colorBitmap) {
        Bitmap grayScaleBitmap = Bitmap.createBitmap(colorBitmap.getWidth(), colorBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        for (int i = 0; i < colorBitmap.getWidth(); i++) {
            for (int j = 0; j < colorBitmap.getHeight(); j++) {
                int p = colorBitmap.getPixel(i, j);
                int r = Color.red(p);
                int g = Color.green(p);
                int b = Color.blue(p);
                int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);
                grayScaleBitmap.setPixel(i, j, Color.rgb(gray, gray, gray));
            }
        }
        return grayScaleBitmap;
    }
    public Mat loadImage(String imgFile) {
        Mat img = Imgcodecs.imread(imgFile, Imgcodecs.IMREAD_COLOR);
        if (img.channels() == 4) {
            Imgproc.cvtColor(img, img, Imgproc.COLOR_BGRA2BGR);
        }
        return img;
    }

    public Mat normalizeMeanVariance(Mat inImg, Scalar mean, Scalar variance) {
        Mat img = inImg.clone();
        Core.subtract(img, new Scalar(mean.val[0] * 255.0, mean.val[1] * 255.0, mean.val[2] * 255.0), img);
        Core.divide(img, new Scalar(variance.val[0] * 255.0, variance.val[1] * 255.0, variance.val[2] * 255.0), img);
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

    // The resize_aspect_ratio function is not directly translatable to Java as it uses specific Python features.
    // You would need to implement this manually in Java.

    public Mat cvt2HeatmapImg(Mat img) {
        img.convertTo(img, CvType.CV_8UC1);
        Imgproc.applyColorMap(img, img, Imgproc.COLORMAP_JET);
        return img;
    }
}