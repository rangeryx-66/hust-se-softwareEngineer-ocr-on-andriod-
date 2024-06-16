package com.example.learnonnx;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.android.Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("opencv_java4");
    }
    private ImageView inputImage;
    private Button upload_image;
    private Button processOCRButton;
    private ActivityResultLauncher<String> getContentLauncher;
    private ActivityResultLauncher<Uri> takePictureLauncher;
    private Uri photoUri;
    private TextView ocr_result;
    private final OrtEnvironment OrtEnv=OrtEnvironment.getEnvironment();
    private OrtSession ortCraftSession;
    private OrtSession ortCRNNSession;
    private Canvas canvas;
    @SuppressLint("UseCompatLoadingForDrawables")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inputImage = findViewById(R.id.imageView);
        processOCRButton = findViewById(R.id.process_ocr);
        upload_image = findViewById(R.id.upload_image);
        ocr_result=findViewById(R.id.OCRResult);
        initOpenCV();
        registerLaunchers();
        OrtSession.SessionOptions ortOpt = new OrtSession.SessionOptions();
        try {
            ortCraftSession=OrtEnv.createSession(readModel("craft"),ortOpt);
            ortCRNNSession=OrtEnv.createSession(readModel("crnn"),ortOpt);
        } catch (OrtException | IOException e) {
            throw new RuntimeException(e);
        }
        upload_image.setOnClickListener(view -> chooseDialog());
        processOCRButton.setOnClickListener(view ->
        {
            try {
                // process OCR
                List<Object> final_result= ProcessOCR.startOCR(getMatFromUri(this, photoUri), OrtEnv, ortCraftSession,ortCRNNSession);
                InputStream inputStream = this.getContentResolver().openInputStream(photoUri);
                Bitmap photo = BitmapFactory.decodeStream(inputStream);
                if(inputStream!=null)
                    inputStream.close();
                Bitmap mutableBitmap = photo.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(mutableBitmap);
                Paint paint= new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(5);
                float xratio= ((float) photo.getWidth()) /800;
                float yratio=((float) photo.getHeight())/608;
                List<int[]> intHorizenlist=(List<int[]>) final_result.get(1);
                for(int[] box:intHorizenlist){
                    canvas.drawRect((float)box[0]*xratio,(float)box[2]*yratio,(float)box[1]*xratio,(float)box[3]*yratio,paint);
                }
                inputImage.setImageBitmap(mutableBitmap);
                ocr_result.setText((String)final_result.get(0));
            } catch (OrtException | IOException e) {
                throw new RuntimeException(e);
            }
        });
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (ortCraftSession != null) {
            try {
                ortCraftSession.close();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }
        if (ortCRNNSession != null) {
            try {
                ortCRNNSession.close();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }
        if (OrtEnv != null) {
            OrtEnv.close();
        }
    }
    private void registerLaunchers() {
        getContentLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        photoUri=uri;
                        inputImage.setImageURI(uri);
                    }
                });

        takePictureLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicture(),
                result -> {
                    if (result) {
                        inputImage.setImageURI(photoUri);
                    }
                });
    }

    private void chooseDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("选择图片");
        builder.setNegativeButton("Gallery", (dialogInterface, i) -> {
            if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 1);
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED)
                    getContentLauncher.launch("image/*");
            } else {
                getContentLauncher.launch("image/*");
            }
        });
        builder.setPositiveButton("Camera", (dialogInterface, i) -> {
            if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    photoUri = createImageUri();
                    if (photoUri != null) {
                        takePictureLauncher.launch(photoUri);
                    }
                }
            } else {
                photoUri = createImageUri();
                if (photoUri != null) {
                    takePictureLauncher.launch(photoUri);
                }
            }
        });
        builder.setNeutralButton("Cancel", (dialogInterface, i) -> dialogInterface.dismiss());
        builder.show();
    }

    private Uri createImageUri() {
        String imageName = "JPEG_" + System.currentTimeMillis() + ".jpg";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, imageName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        return getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
    }

    public static Mat[] getMatFromUri(Context context, Uri uri) {
        Mat mat = null;
        try {
            InputStream inputStream = context.getContentResolver().openInputStream(uri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            if (bitmap != null) {
                mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
                Utils.bitmapToMat(bitmap, mat);
            }
            if (inputStream != null) {
                inputStream.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
                return new Mat[]{mat};
    }
    private void initOpenCV() {
        if (OpenCVLoader.initLocal()) {
            Log.d(TAG, "OpenCV loaded");
        } else {
            Log.e(TAG, "OpenCV not loaded");
        }
    }
    private byte[] readModel(String model) throws IOException {
        int modelID = 0;
        if (Objects.equals(model, "craft"))
            modelID = R.raw.craft;
        if(Objects.equals(model, "crnn"))
            modelID = R.raw.detector;
        InputStream is = MainActivity.this.getResources().openRawResource(modelID);
        return is.readAllBytes();
    }
    private Bitmap Mat32FC1toBitmap(Mat test){
        Mat normalized=new Mat();
        Core.normalize(test,normalized,0,255,Core.NORM_MINMAX);
        Mat unit8=new Mat();
        normalized.convertTo(unit8,CvType.CV_8UC1);
        Log.d(TAG, "onCreate: unit8"+unit8);
        Bitmap testbitmap=Bitmap.createBitmap(unit8.cols(),unit8.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(unit8,testbitmap);
        return testbitmap;
    }

    public static final String TAG = "MyMainTest";
    private File getOutputFile() {
        File directory = getFilesDir();
        File file = new File(directory, "output.txt");
        Log.d("FilePath", "File path: " + file.getAbsolutePath());
        return file;
    }
}
