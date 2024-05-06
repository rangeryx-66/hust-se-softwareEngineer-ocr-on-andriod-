package com.example.myapplication;

import java.util.ArrayList;

public class ImagePostProcessor {

    public ArrayList<float[]> adjustResultCoordinates(ArrayList<float[]> polys, float ratioW, float ratioH, float ratioNet) {
        if (polys.size() > 0) {
            for (int k = 0; k < polys.size(); k++) {
                float[] poly = polys.get(k);
                if (poly != null) {
                    for (int i = 0; i < poly.length; i += 2) {
                        poly[i] *= ratioW * ratioNet;
                        poly[i + 1] *= ratioH * ratioNet;
                    }
                }
            }
        }
        return polys;
    }
}
