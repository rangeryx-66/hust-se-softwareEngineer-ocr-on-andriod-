package com.example.learnonnx;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;
import java.util.Arrays;

public class PostProcess {
    private static final String TAG = "PostProcess";
    public static List<Object> postProcess(float[][][][] scoreTextLink) {
        float text_threshold=0.2f, link_threshold=0.4f, low_text=0.4f;
        boolean poly=false;
        float ratio_w=1,ratio_h=1;
        float[][][] theTextLink = scoreTextLink[0]; // Shape: [304, 400, 2]
        int row=theTextLink.length;
        int column=theTextLink[0].length;
        Mat score_text = new Mat(row, column, CvType.CV_32F);
        Mat score_link = new Mat(row, column, CvType.CV_32F);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                score_text.put(i, j, theTextLink[i][j][0]);
                score_link.put(i, j, theTextLink[i][j][1]);
            }
        }
        List<RotatedRect> boxes = getDetBoxesCore(score_text, score_link, text_threshold, link_threshold, low_text);
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h);
        List<int[]> result=convertPolys(boxes);
        List<Object>ans=new ArrayList<>();

        List<Object>hflist=groupTextBox(result);
        List<double[]>horizontalList=(List<double[]>)hflist.get(0);
        List<double[][]>freeList=(List<double[][]>)hflist.get(1);
        List<double[]> horizontalAgg = new ArrayList<>();
        for (double[] item:horizontalList){
            if (Math.max(item[1] - item[0], item[3] - item[2]) > 20)
                horizontalAgg.add(item);
        }

        List<double[][]> freeListAgg = new ArrayList<>();

        for (double[][] box : freeList) {
            double[] xCoords = new double[box.length];
            double[] yCoords = new double[box.length];

            for (int j = 0; j < box.length; j++) {
                xCoords[j] = box[j][0];
                yCoords[j] = box[j][1];
            }

            double xDiff = diff(xCoords);
            double yDiff = diff(yCoords);

            if (Math.max(xDiff, yDiff) > 20) {
                freeListAgg.add(box);
            }
        }


        ans.add(horizontalAgg);
        ans.add(freeListAgg);
        return ans;
    }
    public static double diff(double[] coords) {
        double min = coords[0];
        double max = coords[0];

        for (double coord : coords) {
            if (coord < min) {
                min = coord;
            }
            if (coord > max) {
                max = coord;
            }
        }

        return max - min;
    }
    private static int[] convertMatOfPoint2fToIntArray(MatOfPoint2f box) {
        int[] poly = new int[box.rows() * box.cols()];
        float[] points = new float[box.rows() * box.cols() * 2];
        box.get(0, 0, points);

        for (int i = 0; i < points.length; i += 2) {
            poly[i] = Math.round(points[i]);
            poly[i + 1] = Math.round(points[i + 1]);
        }

        return poly;
    }
    private static List<RotatedRect> adjustResultCoordinates(List<RotatedRect>polys, float ratio_w, float ratio_h) {
        if (polys == null || polys.isEmpty()) {
            return polys;
        }

        double scaleX = ratio_w * 2;
        double scaleY = ratio_h * 2;

        for (RotatedRect rect : polys) {
            if (rect != null) {
                // Adjust the center point
                rect.center.x *= scaleX;
                rect.center.y *= scaleY;

                // Adjust the size
                rect.size.width *= scaleX;
                rect.size.height *= scaleY;
            }
        }
        return polys;
    }
    public static List<RotatedRect> getDetBoxesCore(Mat rawTextMap, Mat rawLinkMap, double textThreshold, double linkThreshold, double lowText) {
        // prepare data
        Mat linkMap = rawLinkMap.clone();
        Mat textMap = rawTextMap.clone();
        int imgH = textMap.rows();
        int imgW = textMap.cols();
        // labeling method
        Mat textScore = new Mat();
        Imgproc.threshold(textMap, textScore, lowText, 1, Imgproc.THRESH_BINARY);
        Mat linkScore = new Mat();
        Imgproc.threshold(linkMap, linkScore, linkThreshold, 1, Imgproc.THRESH_BINARY);
        Mat textScoreComb = new Mat();
        Core.add(textScore, linkScore, textScoreComb);
        Core.min(textScoreComb, Scalar.all(1), textScoreComb);
        textScoreComb.convertTo(textScoreComb,CvType.CV_8U);
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int nLabels = Imgproc.connectedComponentsWithStats(textScoreComb, labels, stats, centroids,4);
        List<RotatedRect> det = new ArrayList<>();
        List<Integer> mapper = new ArrayList<>();

        for (int k = 1; k < nLabels; k++) {
            Mat kstat=stats.row(k);
            int size = (int) kstat.get(0, Imgproc.CC_STAT_AREA)[0];
            if (size < 10) continue;

            Mat mask = new Mat();
            Core.inRange(labels, new Scalar(k), new Scalar(k), mask);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(textMap, mask);
            if (mmr.maxVal < textThreshold) continue;

            Mat segmap = Mat.zeros(textMap.size(), CvType.CV_8U);
            segmap.setTo(new Scalar(255), mask);
            Mat mask1 = new Mat();
            Mat mask2 = new Mat();
            Mat combinedMask = new Mat();
            Core.compare(linkScore, new Scalar(1), mask1, Core.CMP_EQ);
            Core.compare(textScore, new Scalar(0), mask2, Core.CMP_EQ);
            Core.bitwise_and(mask1, mask2, combinedMask);
            segmap.setTo(new Scalar(0), combinedMask);

            int x = (int) kstat.get(0,Imgproc.CC_STAT_LEFT)[0];
            int y = (int) kstat.get(0,Imgproc.CC_STAT_TOP)[0];
            int w = (int) kstat.get(0,Imgproc.CC_STAT_WIDTH)[0];
            int h = (int) kstat.get(0,Imgproc.CC_STAT_HEIGHT)[0];
            int niter = (int) (Math.sqrt(size * Math.min(w, h) / (double) (w * h)) * 2);

            int sx = Math.max(0, x - niter);
            int ex = Math.min(imgW, x + w + niter + 1);
            int sy = Math.max(0, y - niter);
            int ey = Math.min(imgH, y + h + niter + 1);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1 + niter, 1 + niter));
            Imgproc.dilate(segmap.submat(sy, ey, sx, ex), segmap.submat(sy, ey, sx, ex), kernel);

            List<Point> points = new ArrayList<>();
            for (int i = 0; i < segmap.rows(); i++) {
                for (int j = 0; j < segmap.cols(); j++) {
                    if (segmap.get(i, j)[0] != 0) {
                        points.add(new Point(j, i));
                    }
                }
            }

            MatOfPoint matOfPoint = new MatOfPoint();
            matOfPoint.fromList(points);
            RotatedRect rectangle = Imgproc.minAreaRect(new MatOfPoint2f(matOfPoint.toArray()));

            double wRect = Math.abs(rectangle.size.width);
            double hRect = Math.abs(rectangle.size.height);
            double boxRatio = Math.max(wRect, hRect) / (Math.min(wRect, hRect) + 1e-5);
            if (Math.abs(1 - boxRatio) <= 0.1) {
                int l = points.stream().mapToInt(p -> (int) p.x).min().getAsInt();
                int r = points.stream().mapToInt(p -> (int) p.x).max().getAsInt();
                int t = points.stream().mapToInt(p -> (int) p.y).min().getAsInt();
                int b = points.stream().mapToInt(p -> (int) p.y).max().getAsInt();
                Point[] box = {
                        new Point(l, t),
                        new Point(r, t),
                        new Point(r, b),
                        new Point(l, b)
                };
                rectangle = Imgproc.minAreaRect(new MatOfPoint2f(box));
            }

            Point[] box = new Point[4];
            rectangle.points(box);
            Arrays.sort(box, Comparator.comparingDouble(a -> a.y + a.x));
            Point[] reorderedBox = {box[3], box[0], box[1], box[2]};
            rectangle = Imgproc.minAreaRect(new MatOfPoint2f(reorderedBox));

            det.add(rectangle);
        }

        return det;
    }
    public static List<int[]> convertPolys(List<RotatedRect> polys) {
        List<int[]> singleImgResult = new ArrayList<>();

        for (RotatedRect box : polys) {
            Point[] vertices = new Point[4];
            box.points(vertices);

            int[] intVertices = new int[8];
            for (int i = 0; i < vertices.length; i++) {
                intVertices[2 * i] = (int) Math.round(vertices[i].x);
                intVertices[2 * i + 1] = (int) Math.round(vertices[i].y);
            }

            singleImgResult.add(intVertices);
        }

        return singleImgResult;
    }
    public static List<Object> groupTextBox(List<int[]> polys) {
        double slopeThs=0.1;
        double ycenterThs=0.5;
        double heightThs=0.5;
        double widthThs=1.0; double addMargin=0.05; boolean sortOutput=true;
        List<double[]> horizontalList = new ArrayList<>();
        List<double[][]> freeList = new ArrayList<>();
        List<List<double[]>> combinedList = new ArrayList<>();
        List<double[]> mergedList = new ArrayList<>();

        for (int[] poly : polys) {
            double slopeUp = (double) (poly[3] - poly[1]) / Math.max(10, (poly[2] - poly[0]));
            double slopeDown = (double) (poly[5] - poly[7]) / Math.max(10, (poly[4] - poly[6]));
            if (Math.max(Math.abs(slopeUp), Math.abs(slopeDown)) < slopeThs) {
                double xMax = IntStream.of(0, 2, 4, 6).mapToDouble(i -> poly[i]).max().orElse(0);
                double xMin = IntStream.of(0, 2, 4, 6).mapToDouble(i -> poly[i]).min().orElse(0);
                double yMax = IntStream.of(1, 3, 5, 7).mapToDouble(i -> poly[i]).max().orElse(0);
                double yMin = IntStream.of(1, 3, 5, 7).mapToDouble(i -> poly[i]).min().orElse(0);
                horizontalList.add(new double[]{xMin, xMax, yMin, yMax, 0.5 * (yMin + yMax), yMax - yMin});
            } else {
                double height = Math.hypot(poly[6] - poly[0], poly[7] - poly[1]);
                double width = Math.hypot(poly[2] - poly[0], poly[3] - poly[1]);

                int margin = (int) (1.44 * addMargin * Math.min(width, height));

                double theta13 = Math.abs(Math.atan((double) (poly[1] - poly[5]) / Math.max(10, (poly[0] - poly[4]))));
                double theta24 = Math.abs(Math.atan((double) (poly[3] - poly[7]) / Math.max(10, (poly[2] - poly[6]))));

                double x1 = poly[0] - Math.cos(theta13) * margin;
                double y1 = poly[1] - Math.sin(theta13) * margin;
                double x2 = poly[2] + Math.cos(theta24) * margin;
                double y2 = poly[3] - Math.sin(theta24) * margin;
                double x3 = poly[4] + Math.cos(theta13) * margin;
                double y3 = poly[5] + Math.sin(theta13) * margin;
                double x4 = poly[6] - Math.cos(theta24) * margin;
                double y4 = poly[7] + Math.sin(theta24) * margin;

                freeList.add(new double[][]{{x1, y1}, {x2, y2}, {x3, y3}, {x4, y4}});
            }
        }

        horizontalList.sort(Comparator.comparingDouble(item -> item[4]));

        List<double[]> newBox = new ArrayList<>();
        for (double[] poly : horizontalList) {
            if (newBox.isEmpty()) {
                newBox.add(poly);
            } else {
                double meanHeight = newBox.stream().mapToDouble(b -> b[5]).average().orElse(0);
                double meanYCenter = newBox.stream().mapToDouble(b -> b[4]).average().orElse(0);
                if (Math.abs(meanYCenter - poly[4]) < ycenterThs * meanHeight) {
                    newBox.add(poly);
                } else {
                    combinedList.add(new ArrayList<>(newBox));
                    newBox.clear();
                    newBox.add(poly);
                }
            }
        }
        combinedList.add(new ArrayList<>(newBox));

        for (List<double[]> boxes : combinedList) {
            if (boxes.size() == 1) {
                double[] box = boxes.get(0);
                int margin = (int) (addMargin * Math.min(box[1] - box[0], box[5]));
                mergedList.add(new double[]{box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin});
            } else {
                boxes.sort(Comparator.comparingDouble(item -> item[0]));
                List<List<double[]>> mergedBox = new ArrayList<>();
                newBox.clear();
                for (double[] box : boxes) {
                    if (newBox.isEmpty()) {
                        newBox.add(box);
                    } else {
                        double meanHeight = newBox.stream().mapToDouble(b -> b[5]).average().orElse(0);
                        double xMax = newBox.stream().mapToDouble(b -> b[1]).max().orElse(0);
                        if (Math.abs(meanHeight - box[5]) < heightThs * meanHeight && (box[0] - xMax) < widthThs * (box[3] - box[2])) {
                            newBox.add(box);
                        } else {
                            mergedBox.add(new ArrayList<>(newBox));
                            newBox.clear();
                            newBox.add(box);
                        }
                    }
                }
                if (!newBox.isEmpty()) mergedBox.add(new ArrayList<>(newBox));

                for (List<double[]> mbox : mergedBox) {
                    if (mbox.size() != 1) {
                        double xMin = mbox.stream().mapToDouble(b -> b[0]).min().orElse(0);
                        double xMax = mbox.stream().mapToDouble(b -> b[1]).max().orElse(0);
                        double yMin = mbox.stream().mapToDouble(b -> b[2]).min().orElse(0);
                        double yMax = mbox.stream().mapToDouble(b -> b[3]).max().orElse(0);

                        double boxWidth = xMax - xMin;
                        double boxHeight = yMax - yMin;
                        int margin = (int) (addMargin * Math.min(boxWidth, boxHeight));

                        mergedList.add(new double[]{xMin - margin, xMax + margin, yMin - margin, yMax + margin});
                    } else {
                        double[] box = mbox.get(0);
                        double boxWidth = box[1] - box[0];
                        double boxHeight = box[3] - box[2];
                        int margin = (int) (addMargin * Math.min(boxWidth, boxHeight));

                        mergedList.add(new double[]{box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin});
                    }
                }
            }
        }

        List<Object> result = new ArrayList<>();
        result.add(mergedList);
        result.add(freeList);
        return result;
    }
}
