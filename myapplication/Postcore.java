package com.example.myapplication;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class Postcore {

    public static List<MatOfPoint2f> getDetBoxesCore(Mat textmap, Mat linkmap, double textThreshold, double linkThreshold, double lowText) {
        // prepare data
        Mat linkMapCopy = linkmap.clone();
        Mat textMapCopy = textmap.clone();
        int imgH = textmap.rows();
        int imgW = textmap.cols();

        // labeling method
        Mat textScore = new Mat();
        Imgproc.threshold(textmap, textScore, lowText, 1, 0);
        Mat linkScore = new Mat();
        Imgproc.threshold(linkmap, linkScore, linkThreshold, 1, 0);
        Mat textScoreComb = new Mat();
        Core.add(textScore, linkScore, textScoreComb);
        Core.minMaxLoc(textScoreComb);
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int nLabels = Imgproc.connectedComponentsWithStats(textScoreComb, labels, stats, centroids);
        List<MatOfPoint2f> det = new ArrayList<>();
        List<Integer> mapper = new ArrayList<>();
        for (int k = 1; k < nLabels; k++) {
            // size filtering
            double size = stats.get(k, Imgproc.CC_STAT_AREA)[0];;
            if (size < 10) continue;

            // thresholding
            Mat mask = new Mat();
            Core.compare(labels, new Scalar(k), mask, Core.CMP_EQ);

            // 在满足条件 labels == k 的像素区域中，找到 textmap 的最小和最大像素值
            Core.MinMaxLocResult minMaxResult = Core.minMaxLoc(textmap, mask);

            // 获取最小和最大像素值
            double minValue = minMaxResult.minVal;
            double maxValue = minMaxResult.maxVal;
            if (minValue < textThreshold) {
                // 如果满足条件，执行 continue
                continue;
            }

            // make segmentation map
            Mat segmap = new Mat(textmap.size(), CvType.CV_8U);
            Core.compare(labels, Scalar.all(k), segmap, Core.CMP_EQ);

            mapper.add(k);

            for (int i = 0; i < segmap.rows(); i++) {
                for (int j = 0; j < segmap.cols(); j++) {
                    // 获取 link_score 和 text_score 中的值
                    double linkScoreValue = linkScore.get(i, j)[0];
                    double textScoreValue = textScore.get(i, j)[0];

                    // 检查条件
                    if (linkScoreValue == 1 && textScoreValue == 0) {
                        // 将对应的 segmap 像素设置为 0
                        segmap.put(i, j, 0);
                    }
                }
            }
            int x = (int)stats.get(k,Imgproc.CC_STAT_LEFT)[0];
            int y = (int)stats.get(k,Imgproc.CC_STAT_TOP)[0];
            int w = (int)stats.get(k,Imgproc.CC_STAT_WIDTH)[0];
            int h = (int)stats.get(k,Imgproc.CC_STAT_HEIGHT)[0];
            int niter = (int) Math.sqrt(size * Math.min(w, h) / (w * h)) * 2;
            int sx = x - niter;
            int ex = x + w + niter + 1;
            int sy = y - niter;
            int ey = y + h + niter + 1;
            // boundary check
            sx = Math.max(sx, 0);
            sy = Math.max(sy, 0);
            ex = Math.min(ex, imgW);
            ey = Math.min(ey, imgH);
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1 + niter, 1 + niter));
            Imgproc.dilate(segmap.submat(new Rect(sx, sy, ex - sx, ey - sy)), segmap.submat(new Rect(sx, sy, ex - sx, ey - sy)), kernel);

            // make box
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(segmap, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // 初始化一个列表用于存储检测到的矩形
            List<Point[]> detections = new ArrayList<>();

            // 遍历每个轮廓
            for (MatOfPoint contour : contours) {
                // 将轮廓转换为2D点集
                MatOfPoint2f contour2f = new MatOfPoint2f();
                contour.convertTo(contour2f, CvType.CV_32F);

                // 寻找最小外接矩形
                RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);

                // 获取最小外接矩形的顶点
                Point[] boxPoints = new Point[4];
                rotatedRect.points(boxPoints);

                // 对齐菱形
                // 计算两个向量的范数
                Mat vec1 = new Mat(2, 1, CvType.CV_64FC1);
                vec1.put(0, 0, boxPoints[0].x - boxPoints[1].x);
                vec1.put(1, 0, boxPoints[0].y - boxPoints[1].y);
                double w1 = Core.norm(vec1);

                Mat vec2 = new Mat(2, 1, CvType.CV_64FC1);
                vec2.put(0, 0, boxPoints[1].x - boxPoints[2].x);
                vec2.put(1, 0, boxPoints[1].y - boxPoints[2].y);
                double h1 = Core.norm(vec2);

                double boxRatio = Math.max(w1, h1) / (Math.min(w1, h1) + 1e-5);
                if (Math.abs(1 - boxRatio) <= 0.1) {
                    double l = Double.MAX_VALUE, r = Double.MIN_VALUE, t = Double.MAX_VALUE, b = Double.MIN_VALUE;
                    for (Point point : boxPoints) {
                        l = Math.min(l, point.x);
                        r = Math.max(r, point.x);
                        t = Math.min(t, point.y);
                        b = Math.max(b, point.y);
                    }
                    boxPoints = new Point[]{new Point(l, t), new Point(r, t), new Point(r, b), new Point(l, b)};
                }

                // 顺时针排序
                int startIdx = 0;
                double minSum = Double.MAX_VALUE;
                for (int i = 0; i < 4; i++) {
                    double sum = boxPoints[i].x + boxPoints[i].y;
                    if (sum < minSum) {
                        minSum = sum;
                        startIdx = i;
                    }
                }
                Point[] sortedBoxPoints = new Point[4];
                for (int i = 0; i < 4; i++) {
                    sortedBoxPoints[i] = boxPoints[(startIdx + i) % 4];
                }

                // 将矩形存储到检测列表中
                det.add(new MatOfPoint2f(sortedBoxPoints));
            }


    }return det;}
}