package com.example.myapplication;
import static com.example.myapplication.Recognizer.recognizerPredict;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class Recognizer2 {

    public List<String> recognize(Mat imgCvGrey, List< List<List<Point>>> horizontalList, List< List<List<Point>>> freeList,
                                  String decoder, int beamWidth, int batchSize, int workers,
                                  double contrastThs, double adjustContrast, double filterThs, boolean reformat) throws OrtException {
        List<String> result = new ArrayList<>();
        List<Mat>imageList = new ArrayList<>();
        imageList.add(imgCvGrey);
        if (reformat) {
            // Assuming reformat_input function is implemented
            // imgCvGrey = reformat_input(imgCvGrey);
        }

        String ignoreChar = ""; // Compute ignore_char here

        for (List<List<Point>>bbox : horizontalList) {
            List<Object> processedImages = getImageList1.getImageList(bbox,new ArrayList<>(),imgCvGrey,64,true);
            List<String> result0 = getText((List<Mat>) processedImages.get(0),(int)processedImages.get(1),ignoreChar);
            result.addAll(result0);
        }

        for (List<List<Point>>fbox : freeList) {
            List<Object> processedImages=getImageList1.getImageList(new ArrayList<>(),fbox,imgCvGrey,64,true);
            List<String> result0 = getText((List<Mat>) processedImages.get(0),(int)processedImages.get(1),ignoreChar);
            result.addAll(result0);
        }

        return result;
    }



    private List<String> getText(List<Mat> imageList,int maxWidth, String ignoreChar) throws OrtException {

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("model.onnx", options);
        AlignCollate alignCollate = new AlignCollate(64, maxWidth, false, 1.5);


        Mat[] processedImages = alignCollate.call(imageList);
        List<Mat>picList = new ArrayList<>();
        for(Mat img:processedImages){
            Imgcodecs.imwrite("test.jpg",img);
            picList.add(img);
        }
        // Example usage

        Map<String, String> dictList = new HashMap<>();
        dictList.put("ch_sim", "ch-pin-syl.txt");
        CTCLabelConverter converter = new CTCLabelConverter("0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",new HashMap<>(),dictList);

        List<String[]> results = recognizerPredict(session, converter, picList, maxWidth/10);
        List<String> result = new ArrayList<>();
        for (String[] res : results) {
            for(String s:res){
                result.add(s);
            }
        }
        return result;
    }
}

