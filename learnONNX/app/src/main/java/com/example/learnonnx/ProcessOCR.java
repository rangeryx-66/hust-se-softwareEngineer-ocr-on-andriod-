package com.example.learnonnx;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ProcessOCR {
    public final static String TAG = "ProcessOCR";
    public static List<Object> startOCR(Mat[] images, OrtEnvironment ortEnv, OrtSession ortCraftSession, OrtSession ortCrnnSession) throws OrtException {
        List<Object> thefinalResult = new ArrayList<>();
        StringBuilder finalResult = new StringBuilder();
        float[][][][] scoreTextLink;
        scoreTextLink=TextDetector.testNet(images,ortEnv,ortCraftSession);
        float xratio=((float)images[0].cols())/800;
        float yratio=((float)images[0].rows())/608;
        Log.d(TAG, "startOCR: detector finished");
        List<Object> postResult=PostProcess.postProcess(scoreTextLink);
        Log.d(TAG, "startOCR: postprocess finished");
        List<int[]> intHorizenlist= convertHorizenListToIntList((List<double[]>) postResult.get(0));
        List<int[][]> intFreelist=convertFreeListToIntList((List<double[][]>) postResult.get(1));
        List<List<Point>> horizenlist=convertToPoint1(intHorizenlist,xratio,yratio);
        List<List<Point>> freelist=convertToPoint2(intFreelist);
        List<String>recognizeResult=Recognizer.recognize(images,horizenlist,freelist,ortEnv,ortCrnnSession);
        finalResult= new StringBuilder(String.join(",", recognizeResult));
//        for(int[] horizen:intHorizenlist){
//            finalResult.append(Arrays.toString(horizen)).append("\n");
//        }
//        Log.d(TAG, "startOCR: "+horizenlist.size());
//        Log.d(TAG, "startOCR: "+finalResult.toString());
        thefinalResult.add(finalResult.toString());
        thefinalResult.add(intHorizenlist);
        return thefinalResult;
    }
    public static List<int[]> convertHorizenListToIntList(List<double[]> doubleList) {
        List<int[]> intList = new ArrayList<>();
        for (double[] doubleArray : doubleList) {
            int[] intArray = new int[doubleArray.length];
            for (int i = 0; i < doubleArray.length; i++) {
                intArray[i] = (int) doubleArray[i]; // 强制转换double为int
            }
            intList.add(intArray);
        }
        return intList;
    }
    public static List<int[][]> convertFreeListToIntList(List<double[][]> doubleList) {
        List<int[][]> intList = new ArrayList<>();

        for (double[][] doubleArray : doubleList) {
            int rows = doubleArray.length;
            int cols = doubleArray[0].length;
            int[][] intArray = new int[rows][cols];

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    intArray[i][j] = (int) doubleArray[i][j];
                }
            }

            intList.add(intArray);
        }

        return intList;
    }
    public static List<List<Point>> convertToPoint1(List<int[]> listOfIntArrays,float xratio,float yratio) {

        List<List<Point>> listOfListsOfPoints = new ArrayList<>();

        for (int[] box : listOfIntArrays) {
            List<Point> points = new ArrayList<>();
            points.add(new Point(box[0]*xratio, box[3]*yratio)); // 左上角
            points.add(new Point(box[1]*xratio, box[3]*yratio)); // 右上角
            points.add(new Point(box[1]*xratio, box[2]*yratio)); // 右下角
            points.add(new Point(box[0]*xratio, box[2]*yratio)); // 左下角
            listOfListsOfPoints.add(points);
        }

        return listOfListsOfPoints;
    }
    public static List<List<Point>> convertToPoint2(List<int[][]> listOfIntArrays) {

        List<List<Point>> listOfListsOfPoints = new ArrayList<>();

        for (int[][] box : listOfIntArrays) {
            List<Point> points = new ArrayList<>();
            for (int[] point : box){
                points.add(new Point(point[0], point[1]));
            }
            listOfListsOfPoints.add(points);
        }
        return listOfListsOfPoints;
    }


}