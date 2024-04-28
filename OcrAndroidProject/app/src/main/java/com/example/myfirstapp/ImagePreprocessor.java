import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.android.Utils;
import org.opencv.core.Size;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.content.Context;

public class ImagePreprocessor {

    public Mat loadImage(String imagePath, Context context) {
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        if (bitmap == null) {
            // 尝试从资源中加载
            int imageID = context.getResources().getIdentifier(imagePath, "drawable", context.getPackageName());
            bitmap = BitmapFactory.decodeResource(context.getResources(), imageID);
        }
        Mat mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        // 转换为 RGB，如果需要
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB);
        return mat;
    }
    public Mat normalizeMeanVariance(Mat input) {
        Scalar means = new Scalar(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0);
        Scalar stds = new Scalar(0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0);

        input.convertTo(input, CvType.CV_32F);
        Core.subtract(input, means, input);
        Core.divide(input, stds, input);
        return input;
    }
    public Mat resizeAspectRatio(Mat img, int squareSize, int interpolation, double magRatio) {
        int width = img.width();
        int height = img.height();
        int maxDimension = Math.max(width, height);
        double targetSize = magRatio * maxDimension;

        if (targetSize > squareSize) {
            targetSize = squareSize;
        }

        double ratio = targetSize / maxDimension;
        Size newSize = new Size(width * ratio, height * ratio);

        Mat resizedImage = new Mat();
        Imgproc.resize(img, resizedImage, newSize, 0, 0, interpolation);

        return resizedImage;
    }
    public Mat convertToHeatmap(Mat img) {
        Core.normalize(img, img, 0, 255, Core.NORM_MINMAX);
        Imgproc.applyColorMap(img, img, Imgproc.COLORMAP_JET);
        return img;
    }

}

