package com.example.learnonnx;
import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;


import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class AlignCollate {

    private int imgH;
    private int imgW;
    private boolean keepRatioWithPad;
    private double adjustContrast;
    public final static String TAG = "AlignCollate";

    public AlignCollate(int imgH, int imgW, boolean keepRatioWithPad,double adjustContrast) {
        this.imgH = imgH;
        this.imgW = imgW;
        this.keepRatioWithPad = keepRatioWithPad;
        this.adjustContrast = adjustContrast;
    }

    public List<FloatBuffer> call(List<Mat> batch) {
        List<FloatBuffer> resizedImages = new ArrayList<>();

        for (int k =0;k<batch.size();k++) {
            Mat image=batch.get(k);
            if (image == null) {
                continue;
            }

            int w = image.width();
            int h = image.height();

            // Adjust contrast if necessary
            if (adjustContrast > 0) {
                image.convertTo(image, -1, adjustContrast, 0);
            }
            if (h==imgH){
                Mat resizedImage = new Mat();
                Size newSize = new Size(100, 32);
                Imgproc.resize(image, resizedImage, newSize, 0, 0, Imgproc.INTER_LANCZOS4);
                float []imgData=Recognizer.matToFloatArray(resizedImage);
                for (int i = 0; i < imgData.length; i++) {
                    imgData[i] /= 255.0f;
                    imgData[i] = (imgData[i] - 0.5f) / 0.5f;
                }
                if(k==1)
                    logFloatArray(imgData);
                FloatBuffer buffer=FloatBuffer.wrap(imgData);
                resizedImages.add(buffer);
            }
            else {
                double ratio = (double) w / h;
                int resizedW = (int) (imgH * ratio);
                if (resizedW > imgW) {
                    resizedW = imgW;
                }

                Mat resizedImage = new Mat();
                Size newSize = new Size(resizedW, imgH);
                Imgproc.resize(image, resizedImage, newSize, 0, 0, Imgproc.INTER_CUBIC);

                FloatBuffer normalizedImage = normalizePad(resizedImage, imgW, imgH);
                resizedImages.add(normalizedImage);
            }
        }

        return resizedImages;
    }

    private FloatBuffer normalizePad(Mat img,int imgW,int imgH) {
        int w = img.width();
        int h = img.height();
        int c = img.channels();
        FloatBuffer buffer = FloatBuffer.allocate(imgW*imgH*c);
        float []imgData=Recognizer.matToFloatArray(img);
        for (int i = 0; i < imgData.length; i++) {
            imgData[i] = imgData[i]/255.0f;
            imgData[i] = (imgData[i] - 0.5f) / 0.5f;
        }
        int index = 0;
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    buffer.put(imgData[index++]);
                }
            }
        }
        int paddingSize = imgH - w;
        if (paddingSize > 0) {
            for (int i = 0; i < c; i++) {
                for (int j = 0; j < h; j++) {
                    float padValue = imgData[(c * (h * w)) - 1];
                    for (int k = 0; k < paddingSize; k++) {
                        buffer.put(padValue);
                    }
                }
            }
        }
        buffer.rewind();
        return buffer;
    }
    public static void logFloatArray(float[] floatArray) {
        int maxLogSize = 3000;
        StringBuilder sb = new StringBuilder();
        sb.append("Float array: [");

        for (int i = 0; i < floatArray.length; i++) {
            sb.append(floatArray[i]);
            if (i < floatArray.length - 1) {
                sb.append(", ");
            }
        }

        sb.append("]");

        String finalString = sb.toString();

        for (int i = 0; i < finalString.length(); i += maxLogSize) {
            int end = Math.min(finalString.length(), i + maxLogSize);
            Log.d(TAG, finalString.substring(i, end));
        }
    }
}
