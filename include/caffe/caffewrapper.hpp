/*******************************************************************************
 * Copyright 2018  JD.Inc
 *******************************************************************************/

#ifndef CAFFE_WRAPPER_H
#define CAFFE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif 

struct DetectResult {
    int   labelIdx;
    float score;
    int left;
    int top;
    int right;
    int bottom;
};

// init 
void* InitCaffe();
int ConfigureCaffe(void* handle, const char* modelFile, const char* weightsFile);
int Detect(void* handle, const char* imageFile, const int size, DetectResult* result);
void DestroyCaffe(void* handle);

#ifdef __cplusplus
}
#endif 

#endif //CAFFE_WRAPPER_H