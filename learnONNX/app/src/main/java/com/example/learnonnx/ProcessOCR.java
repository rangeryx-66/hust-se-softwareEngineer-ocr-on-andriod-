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
        List<int[]> intHorizenlist= convertDoubleListToIntList((List<double[]>) postResult.get(0));
//        drawRects(photo,intHorizenlist,xratio,yratio,inputImage);
        List<int[][]> intFreelist=convertToIntList((List<double[][]>) postResult.get(1));
        List<List<Point>> horizenlist=convertToPoint1(intHorizenlist);
        List<List<Point>> freelist=convertToPoint2(intFreelist);
        List<String>recognizeResult=Recognizer.recognize(images,horizenlist,freelist,ortEnv,ortCrnnSession);
        finalResult= new StringBuilder(String.join(",", recognizeResult));
//        for(int[] horizen:intHorizenlist){
//            finalResult.append(Arrays.toString(horizen)).append("\n");
//        }
//        Log.d(TAG, "startOCR: "+horizenlist.size());
//        Log.d(TAG, "startOCR: "+finalResult.toString());
        Log.d(TAG, "startOCR: recognize finished");
        Log.d(TAG, "startOCR: "+finalResult.toString().length());
        thefinalResult.add(finalResult.toString().substring(1,10000));
        thefinalResult.add(intHorizenlist);
        return thefinalResult;
    }
    public static List<int[]> convertDoubleListToIntList(List<double[]> doubleList) {
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
    public static List<int[][]> convertToIntList(List<double[][]> doubleList) {
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
    public static List<List<Point>> convertToPoint1(List<int[]> listOfIntArrays) {

        List<List<Point>> listOfListsOfPoints = new ArrayList<>();

        for (int[] box : listOfIntArrays) {
            List<Point> points = new ArrayList<>();
            Log.w(TAG, "convertToPoint1: "+box[0]+" "+box[1]+" "+box[2]+" "+box[3]);
            points.add(new Point(box[0], box[3])); // 左上角
            points.add(new Point(box[1], box[3])); // 右上角
            points.add(new Point(box[1], box[2])); // 右下角
            points.add(new Point(box[0], box[2])); // 左下角
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
    private static void drawRects(Bitmap photo, List<int[]> intHorizenlist, float xratio, float yratio,ImageView inputImage) {
        Bitmap tempBitmap = photo.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(tempBitmap);
        Paint paint= new Paint();
        paint.setColor(Color.red(255));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(10);
        for(int[] box:intHorizenlist){
            canvas.drawRect((float)box[0]*xratio,(float)box[3]*yratio,(float)box[1]*xratio,(float)box[2]*yratio,paint);
        }
        Log.d(TAG, "drawRects: ");
        inputImage.setImageBitmap(tempBitmap);
    }


}
