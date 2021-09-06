// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.xiaomi.mace.demo.result;

import android.os.Environment;

import java.io.File;

public class InitData {

    public static final String[] DEVICES = new String[]{"CPU", "GPU"};
//    public static final String[] DEVICES = new String[]{"CPU"};

//    public static final String[] MODELS = new String[]{"cifar10_keras_network_1"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"cifar10_keras_network_1"};
//    public static final String[] MODELS = new String[]{"cifar10_keras_network_1_INT8"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"cifar10_keras_network_1_INT8"};
//    public static final String[] MODELS = new String[]{"cifar10_keras_network_4"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"cifar10_keras_network_4"};
//    public static final String[] MODELS = new String[]{"cifar10_keras_network_2_INT8"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"cifar10_keras_network_2_INT8"};


//    public static final String[] MODELS = new String[]{"mnist_keras"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"mnist_keras"};
    public static final String[] MODELS = new String[]{"mnist_keras_network_2"};
    public static final String[] ONLY_CPU_MODELS = new String[]{"mnist_keras_network_2"};
//    public static final String[] MODELS = new String[]{"har-cnn", "kws_tc_resnet8"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"har-cnn", "kws_tc_resnet8"};
//    public static final String[] MODELS = new String[]{"har-cnn"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"har-cnn"};
//    public static final String[] MODELS = new String[]{"kws_tc_resnet8"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"kws_tc_resnet8"};
//    public static final String[] MODELS = new String[]{"mnist_tf_1"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"mnist_tf_1"};
//    public static final String[] MODELS = new String[]{"caffee_mnist"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"caffee_mnist"};
//    public static final String[] MODELS = new String[]{"caffee_cifar10"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"caffee_cifar10"};
//    public static final String[] MODELS = new String[]{"mnist_onnx"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"mnist_onnx"};
//    public static final String[] MODELS = new String[]{"har-cnn", "kws_tc_resnet8", "mnist_tf_1", "caffee_mnist", "caffee_cifar10", "mnist_onnx"};
//    public static final String[] ONLY_CPU_MODELS = new String[]{"har-cnn", "kws_tc_resnet8", "mnist_tf_1", "caffee_mnist", "caffee_cifar10", "mnist_onnx"};

//    public static final String[] MODELS = new String[]{"mobilenet_v1", "mobilenet_v2", "mobilenet_v1_quant", "mobilenet_v2_quant"};
//    private static final String[] ONLY_CPU_MODELS = new String[]{"mobilenet_v1_quant", "mobilenet_v2_quant"};

    private String model;
    private String device = "";
    private String device_GPU = "";
    private String device_thread3 = "";
    private int ompNumThreads;
    private int cpuAffinityPolicy;
    private int openclCacheReusePolicy;
    private int gpuPerfHint;
    private int gpuPriorityHint;
    private String openclCacheFullPath = "";
    private String openclCacheFullPath_GPU = "";
    private String openclCacheFullPathThread3 = "";
    private String storagePath = "";
    private String storagePath_GPU = "";
    private String storagePathThread3 = "";

    public InitData() {
        model = MODELS[0];
        ompNumThreads = 2; //2;
        cpuAffinityPolicy = 1;
        openclCacheReusePolicy = 1;
        gpuPerfHint = 3;
        gpuPriorityHint = 3;
        device = DEVICES[0];
        device_GPU = DEVICES[0];
        device_thread3 = DEVICES[0];
        storagePath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "mace";
        storagePath_GPU = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "mace_GPU";
        storagePathThread3 = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "maceThread3";
        File file = new File(storagePath);
        if (!file.exists()) {
            file.mkdir();
        }
        File file_GPU = new File(storagePath_GPU);
        if (!file_GPU.exists()) {
            file_GPU.mkdir();
        }
        File fileThread3 = new File(storagePathThread3);
        if (!fileThread3.exists()) {
            fileThread3.mkdir();
        }
        openclCacheFullPath  = storagePath + File.separator + "mace_cl_compiled_program.bin";
        openclCacheFullPath_GPU  = storagePath_GPU + File.separator + "mace_cl_compiled_program.bin";
        openclCacheFullPathThread3  = storagePath_GPU + File.separator + "mace_cl_compiled_program.bin";
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public int getOmpNumThreads() {
        return ompNumThreads;
    }

    public void setOmpNumThreads(int ompNumThreads) {
        this.ompNumThreads = ompNumThreads;
    }

    public int getCpuAffinityPolicy() {
        return cpuAffinityPolicy;
    }

    public void setCpuAffinityPolicy(int cpuAffinityPolicy) {
        this.cpuAffinityPolicy = cpuAffinityPolicy;
    }

    public int getOpenclCacheReusePolicy() {
        return openclCacheReusePolicy;
    }

    public void setOpenclCacheReusePolicy(int openclCacheReusePolicy) {
        this.openclCacheReusePolicy = openclCacheReusePolicy;
    }

    public int getGpuPerfHint() {
        return gpuPerfHint;
    }

    public void setGpuPerfHint(int gpuPerfHint) {
        this.gpuPerfHint = gpuPerfHint;
    }

    public int getGpuPriorityHint() {
        return gpuPriorityHint;
    }

    public void setGpuPriorityHint(int gpuPriorityHint) {
        this.gpuPriorityHint = gpuPriorityHint;
    }

    public String getOpenclCacheFullPath() {
        return openclCacheFullPath;
    }

    public void setOpenclCacheFullPath(String openclCacheFullPath) {
        this.openclCacheFullPath = openclCacheFullPath;
    }

    public String getStoragePath() {
        return storagePath;
    }

    public void setStoragePath(String storagePath) {
        this.storagePath = storagePath;
    }

    public static String getCpuDevice() {
        return DEVICES[0];
    }

    public static boolean isOnlySupportCpuByModel(String model) {
        for (String m : ONLY_CPU_MODELS) {
            if (m.equals(model)) {
                return true;
            }
        }
        return false;
    }

    public String getDevice_GPU() {
        return device_GPU;
    }

    public String getDeviceThread3() {
        return device_thread3;
    }

    public String getOpenclCacheFullPath_GPU() { return openclCacheFullPath_GPU;
    }

    public String getStoragePath_GPU() { return storagePath_GPU;
    }
    public String getOpenclCacheFullPathThread3() { return openclCacheFullPathThread3;
    }

    public String getStoragePathThread3() { return storagePathThread3;
    }
}
