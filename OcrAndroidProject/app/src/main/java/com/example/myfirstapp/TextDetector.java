//import org.opencv.core.*;
//import org.opencv.imgproc.Imgproc;
//
//import java.util.ArrayList;
//import java.util.List;
//
//public class TestNet {
//
//    public static List<List<Rect>> testNet(int canvasSize, float magRatio, Net net, Mat[] images,
//                                           float textThreshold, float linkThreshold, boolean lowText, boolean poly,
//                                           boolean estimateNumChars) {
//        List<List<Rect>> boxesList = new ArrayList<>();
//        List<List<Rect>> polysList = new ArrayList<>();
//
//        float ratioH, ratioW;
//        for (Mat image : images) {
//            List<Mat> imgResizedList = new ArrayList<>();
//
//            // Resize
//            Mat imgResized = new Mat();
//            float[] sizeHeatmap = new float[2];
//            resizeAspectRatio(image, imgResized, sizeHeatmap, canvasSize, magRatio);
//            imgResizedList.add(imgResized);
//            ratioH = ratioW = 1 / sizeHeatmap[1];
//
//            // Preprocessing
//            List<Mat> x = new ArrayList<>();
//            for (Mat nImg : imgResizedList) {
//                Mat normalizedImg = normalizeMeanVariance(nImg);
//                Mat transposedImg = new Mat();
//                Core.transpose(normalizedImg, transposedImg);
//                x.add(transposedImg);
//            }
//
//            // Forward pass
//            Mat[] xArray = x.toArray(new Mat[0]);
//            Mat xConcatenated = new Mat();
//            Core.hconcat(xArray, xConcatenated);
//            Tensor tensor = Tensor.fromBlob(xConcatenated.dataAddr(), new long[]{1, xConcatenated.channels(),
//                    xConcatenated.rows(), xConcatenated.cols()}, FloatBuffer.allocate(0));
//            net.setInput(tensor);
//            Mat y = net.forward();
//
//            List<Mat> scoreTextList = new ArrayList<>();
//            List<Mat> scoreLinkList = new ArrayList<>();
//
//            // Split channels
//            Core.split(y, scoreTextList);
//            Core.split(y, scoreLinkList);
//
//            // Post-processing
//            List<Rect> boxes = new ArrayList<>();
//            List<Rect> polys = new ArrayList<>();
//            for (int i = 0; i < scoreTextList.size(); i++) {
//                Mat scoreText = scoreTextList.get(i);
//                Mat scoreLink = scoreLinkList.get(i);
//
//                List<Rect> result = getDetBoxes(scoreText, scoreLink, textThreshold, linkThreshold, lowText, poly,
//                        estimateNumChars, ratioW, ratioH);
//                boxes.addAll(result);
//                polys.addAll(result);
//            }
//            boxesList.add(boxes);
//            polysList.add(polys);
//        }
//
//        return boxesList;
//    }
//
//    private static void resizeAspectRatio(Mat src, Mat dst, float[] targetRatio, int canvasSize, float magRatio) {
//        // Implement resizing aspect ratio here
//    }
//
//    private static Mat normalizeMeanVariance(Mat img) {
//        // Implement normalization here
//        return img;
//    }
//
//    private static List<Rect> getDetBoxes(Mat scoreText, Mat scoreLink, float textThreshold, float linkThreshold,
//                                          boolean lowText, boolean poly, boolean estimateNumChars,
//                                          float ratioW, float ratioH) {
//        // Implement post-processing here
//        return new ArrayList<>();
//    }
//}
