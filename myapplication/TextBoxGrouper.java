package com.example.myapplication;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.opencv.core.MatOfPoint;

public class TextBoxGrouper {

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
            double slopeUp = (poly[3] - poly[1]) / Math.max(10, (poly[2] - poly[0]));
            double slopeDown = (poly[5] - poly[7]) / Math.max(10, (poly[4] - poly[6]));
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

                double theta13 = Math.abs(Math.atan((poly[1] - poly[5]) / Math.max(10, (poly[0] - poly[4]))));
                double theta24 = Math.abs(Math.atan((poly[3] - poly[7]) / Math.max(10, (poly[2] - poly[6]))));

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

        if (sortOutput) {
            horizontalList.sort(Comparator.comparingDouble(item -> item[4]));
        }

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


