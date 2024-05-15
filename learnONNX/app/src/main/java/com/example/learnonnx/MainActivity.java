package com.example.learnonnx;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.*;
import org.opencv.android.Utils;


import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
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
    OrtSession ortSes;
    //    Button processOCRButton = findViewById(R.id.process_ocr);
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
            ortSes=OrtEnv.createSession(readModel(),ortOpt);
        } catch (OrtException | IOException e) {
            throw new RuntimeException(e);
        }
        upload_image.setOnClickListener(view -> chooseDialog());
        processOCRButton.setOnClickListener(view ->
        {
            try {
                getMatFromUri(this, photoUri);
                float[][][][] final_result= TextDetector.testNet(getMatFromUri(this, photoUri), OrtEnv, ortSes);
                Log.w(TAG, "onCreate: result:"+ Arrays.deepToString(final_result));
                Log.w(TAG, "onCreate: the length of result:"+ Arrays.deepToString(final_result).length());
                 ocr_result.setText(Arrays.deepToString(final_result).substring(0,300000));
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        });
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
            // 从 Uri 获取 InputStream
            InputStream inputStream = context.getContentResolver().openInputStream(uri);
            // 将 InputStream 转换为 Bitmap
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            if (bitmap != null) {
                // 创建与 Bitmap 同样大小的 Mat
                mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
                Log.d(TAG, "15 ");
                // 将 Bitmap 转换为 Mat
                Utils.bitmapToMat(bitmap, mat);
                Log.d(TAG, "2 ");
            }
            if (inputStream != null) {
                inputStream.close(); // 关闭 InputStream
                Log.d(TAG, "3 ");
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
    private byte[] readModel() throws IOException {
        int modelID = R.raw.detector_craft;
        InputStream is = MainActivity.this.getResources().openRawResource(modelID);
        return is.readAllBytes();
    }

    public static final String TAG = "MyMainTest";
}

