package com.example.learnonnx;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.opencv.core.*;
import org.opencv.android.Utils;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;

import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
//    private final OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();
//    private OrtSession ortSession;
    private ImageView inputImage;
    private Button upload_image;
    private Button processOCRButton;
    private ActivityResultLauncher<String> getContentLauncher;
    private ActivityResultLauncher<Uri> takePictureLauncher;
    private Uri photoUri;
//    Button processOCRButton = findViewById(R.id.process_ocr);
    @SuppressLint("UseCompatLoadingForDrawables")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inputImage = findViewById(R.id.imageView);
        processOCRButton=findViewById(R.id.process_ocr);
        upload_image = findViewById(R.id.upload_image);
        registerLaunchers();
        upload_image.setOnClickListener(view -> chooseDialog());
        processOCRButton.setOnClickListener(view -> {
         //   TextDetector.testNet(2560,1.f,getMatFromUri(this,photoUri));
        });

//        inputImage.setImageBitmap(BitmapFactory.decodeStream(readInputImage()));
//
//        // Initialize Ort Session and register the ONNX Runtime extensions package
//        // that contains the custom operators.
//        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
//        try {
//            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
//            ortSession = ortEnv.createSession(readModel(), sessionOptions);
//        } catch (Exception e) {
//            Log.e(TAG, "Error creating ONNX session", e);
//        }
//
//        processOCRButton.setOnClickListener(view -> {
//            try {
//                performSuperResolution(ortSession);
//                Toast.makeText(getBaseContext(), "Super resolution performed!", Toast.LENGTH_SHORT).show();
//            } catch (Exception e) {
//                Log.e(TAG, "Exception caught when performing super resolution", e);
//                Toast.makeText(getBaseContext(), "Failed to perform super resolution", Toast.LENGTH_SHORT).show();
//            }
//        });
    }
    private void registerLaunchers() {
        getContentLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
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
    private void chooseDialog(){
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("选择图片");
        builder.setPositiveButton("Gallery", (dialogInterface, i) -> {
            if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED){
                requestPermissions(new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 1);
            } else {
                getContentLauncher.launch("image/*");
            }
        });
        builder.setNegativeButton("Camera", (dialogInterface, i) -> {
            if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            } else {
                photoUri = createImageUri();
                if (photoUri != null) {
                    takePictureLauncher.launch(photoUri);
                }
            }
        });
        builder.show();
    }
    private Uri createImageUri(){
        String imageName = "JPEG_" + System.currentTimeMillis() + ".jpg";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, imageName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        return getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
    }
    public static Mat[] getMatFromUri(Context context, Uri uri) {
        Mat mat = null;
        Mat[] matArray=new Mat[1];
        try {
            // 从 Uri 获取 InputStream
            InputStream inputStream = context.getContentResolver().openInputStream(uri);
            // 将 InputStream 转换为 Bitmap
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            if (bitmap != null) {
                // 创建与 Bitmap 同样大小的 Mat
                mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC4);
                // 将 Bitmap 转换为 Mat
                Utils.bitmapToMat(bitmap, mat);
            }
            if (inputStream != null) {
                inputStream.close(); // 关闭 InputStream
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        matArray[0]=mat;
        return matArray;
    }
//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        if (ortSession != null) {
//            try {
//                ortSession.close();
//            } catch (OrtException e) {
//                throw new RuntimeException(e);
//            }
//        }
//        if (ortEnv != null) {
//            ortEnv.close();
//        }
//    }
//
//    private void updateUI(Bitmap outputBitmap) {
//        outputImage.setImageBitmap(outputBitmap);
//    }
//
//    private byte[] readModel() {
//        int modelID = R.raw.pytorch_superresolution;
//        InputStream is = getResources().openRawResource(modelID);
//        try {
//            return is.readAllBytes();
//        } catch (Exception e) {
//            Log.e(TAG, "Failed to read model", e);
//            return new byte[0];
//        }
//    }
//
//    private InputStream readInputImage() {
//        try {
//            return getAssets().open("super_res_input.png");
//        } catch (Exception e) {
//            Log.e(TAG, "Failed to read input image", e);
//            return null;
//        }
//    }
//
//    private void performSuperResolution(OrtSession ortSession) {
//        SuperResPerformer superResPerformer = new SuperResPerformer();
//        Bitmap result = superResPerformer.upscale(readInputImage(), ortEnv, ortSession).getOutputBitmap();
//        updateUI(result);
//    }
//
//    private static final String TAG = "ORTSuperResolution";
}
