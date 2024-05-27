package com.example.learnonnx;

import android.util.Log;

import org.opencv.core.Mat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ProcessOCR {
    public final static String TAG = "ProcessOCR";
    public static String startOCR(Mat[] images, OrtEnvironment ortEnv, OrtSession ortCraftSession,OrtSession ortCrnnSession) throws OrtException {
        StringBuilder finalResult = new StringBuilder();
        float[][][][] scoreTextLink;

        scoreTextLink=TextDetector.testNet(images,ortEnv,ortCraftSession);
        Log.d(TAG, "startOCR: detector finished");
        List<Object> postResult=PostProcess.postProcess(scoreTextLink);
        Log.d(TAG, "startOCR: postprocess finished");
        List<int[]>horizenlist= convertDoubleListToIntList((List<double[]>) postResult.get(0));
        for(int[] horizen:horizenlist){
            finalResult.append(Arrays.toString(horizen)).append("\n");
        }
        Log.d(TAG, "startOCR: "+horizenlist.size());
        return finalResult.toString();
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
}
