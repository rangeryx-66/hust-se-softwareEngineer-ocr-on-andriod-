package com.example.learnonnx;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Map;

class Result {
    private Bitmap outputBitmap;

    public Result() {
        this.outputBitmap = null;
    }

    public Bitmap getOutputBitmap() {
        return outputBitmap;
    }

    public void setOutputBitmap(Bitmap outputBitmap) {
        this.outputBitmap = outputBitmap;
    }
}

class SuperResPerformer {

    public Result upscale(InputStream inputStream, OrtEnvironment ortEnv, OrtSession ortSession) {
        Result result = new Result();

        try {
            // Step 1: Convert image into byte array (raw image bytes)
            byte[] rawImageBytes = inputStream.readAllBytes(); // Java 9 and above

            // Step 2: Get the shape of the byte array and make ORT tensor
            long[] shape = new long[]{(long) rawImageBytes.length};

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    ortEnv,
                    ByteBuffer.wrap(rawImageBytes),
                    shape,
                    OnnxJavaType.UINT8
            );

            try {
                // Step 3: Call ORT Session run
                Map<String, OnnxTensor> inputs = Collections.singletonMap("image", inputTensor);
                OrtSession.Result output = ortSession.run(inputs);

                try {
                    // Step 4: Output analysis
                    byte[] rawOutput = (byte[]) output.get(0).getValue();
                    Bitmap outputImageBitmap = byteArrayToBitmap(rawOutput);

                    // Step 5: Set output result
                    result.setOutputBitmap(outputImageBitmap);
                } finally {
                    if (output != null) {
                        output.close();
                    }
                }
            } finally {
                if (inputTensor != null) {
                    inputTensor.close();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return result;
    }

    private Bitmap byteArrayToBitmap(byte[] data) {
        return BitmapFactory.decodeByteArray(data, 0, data.length);
    }
}