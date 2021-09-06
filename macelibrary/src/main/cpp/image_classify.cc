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

#include "image_classify.h"

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#include "mace/public/mace.h"
#include "mace/public/mace_engine_factory.h"

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-lib", __VA_ARGS__))

int cumulative_time = 0;
int cumulative_time_CPU = 0;
int cumulative_time_GPU = 0;
int cumulative_time_thread3 = 0;
int frame_no = 0;


namespace {

struct ModelInfo {
  std::string input_name;
  std::string output_name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
};


struct MaceContext {
    std::shared_ptr<mace::OpenclContext> gpu_context;
    std::shared_ptr<mace::MaceEngine> engine;
    std::string model_name;
    mace::RuntimeType runtime_type = mace::RuntimeType::RT_CPU;
    std::map<std::string, ModelInfo> model_infos = {
//            {"cifar10_keras_network_1", {"conv2d_input:0", "activation/Softmax:0", {1, 32, 32, 3}, {1, 10}}}
//            {"cifar10_keras_network_1_INT8", {"conv2d_input_1:0", "quant_activation/Softmax:0", {1, 32, 32, 3}, {1, 10}}}
//            {"cifar10_keras_network_2", {"conv2d_input_2:0", "activation/Softmax_1:0", {1, 32, 32, 3}, {1, 10}}}
//            {"cifar10_keras_network_2_INT8", {"conv2d_input_3:0", "quant_activation/Softmax_1:0", {1, 32, 32, 3}, {1, 10}}}
//            {"cifar10_keras_network_3", {"conv2d_input:0", "activation/Softmax:0", {1, 32, 32, 3}, {1, 10}}}
//            {"cifar10_keras_network_4", {"conv2d_26_input:0", "activation/Softmax:0", {1, 32, 32, 3}, {1, 10}}}

//            {"mnist_keras", {"input_input:0", "output/Softmax:0", {1, 28, 28, 1}, {1, 10}}}
            {"mnist_keras_network_2", {"input_input_1:0", "output/Softmax_1:0", {1, 28, 28, 1}, {1, 10}}}
//            {"har-cnn", {"conv1d/conv1d/ExpandDims", "dense/BiasAdd", {1, 1, 128, 9}, {1, 6}}},
//            {"kws_tc_resnet8", {"input", "output/softmax", {1, 98, 40, 1}, {1, 12}}},
//            {"mnist_tf_1", {"x_2", "output", {1, 28, 28, 1}, {1, 10}}},
//            {"caffee_mnist", {"data", "prob", {1, 28, 28, 1}, {1, 10}}},
//            {"caffee_cifar10", {"data", "prob", {1, 3, 32, 32}, {10, 1}}},
//            {"mnist_onnx", {"input.1", "LogSoftmax_9", {1, 28, 28, 1}, {1, 10}}} //pytorch onnx one

    };
};

struct MaceContext_thread {
    std::shared_ptr<mace::OpenclContext> gpu_context_thread;
    std::shared_ptr<mace::MaceEngine> engine_thread;
    std::string model_name_thread;
    mace::RuntimeType runtime_type_thread = mace::RuntimeType::RT_CPU;
    std::map<std::string, ModelInfo> model_infos_thread = {
//        {"mnist_keras", {"input_input:0", "output/Softmax:0", {1, 28, 28, 1}, {1, 10}}}
        {"mnist_keras_network_2", {"input_input_1:0", "output/Softmax_1:0", {1, 28, 28, 1}, {1, 10}}}
    };
};

struct MaceContextThread3 {
    std::shared_ptr<mace::OpenclContext> gpu_context_thread;
    std::shared_ptr<mace::MaceEngine> engine_thread;
    std::string model_name_thread;
    mace::RuntimeType runtime_type_thread = mace::RuntimeType::RT_CPU;
    std::map<std::string, ModelInfo> model_infos_thread = {
//        {"mnist_keras", {"input_input:0", "output/Softmax:0", {1, 28, 28, 1}, {1, 10}}}
        {"mnist_keras_network_2", {"input_input_1:0", "output/Softmax_1:0", {1, 28, 28, 1}, {1, 10}}}
    };
};

mace::RuntimeType ParseDeviceType(const std::string &device) {
  if (device.compare("CPU") == 0) {
    return mace::RuntimeType::RT_CPU;
  } else if (device.compare("GPU") == 0) {
    return mace::RuntimeType::RT_OPENCL;
  } else if (device.compare("HEXAGON") == 0) {
    return mace::RuntimeType::RT_HEXAGON;
  } else {
    return mace::RuntimeType::RT_CPU;
  }
}

MaceContext& GetMaceContext() {
  // TODO(yejianwu): In multi-dlopen process, this step may cause memory leak.
  static auto *mace_context = new MaceContext;

  return *mace_context;
}

MaceContext_thread& GetMaceContextThread() {
    // TODO(yejianwu): In multi-dlopen process, this step may cause memory leak.
    static auto *mace_context_thread = new MaceContext_thread;

    return *mace_context_thread;
}

MaceContextThread3& GetMaceContextThread3() {
    // TODO(yejianwu): In multi-dlopen process, this step may cause memory leak.
    static auto *mace_context_thread3 = new MaceContextThread3;

    return *mace_context_thread3;
}


}  // namespace

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateGPUContext(
        JNIEnv *env, jclass thisObj, jstring storage_path,
        jstring opencl_cache_full_path, jint opencl_cache_reuse_policy) {
    MaceContext &mace_context = GetMaceContext();
    // DO NOT USE tmp directory.
    // Please use APP's own directory and make sure the directory exists.
    const char *storage_path_ptr = env->GetStringUTFChars(storage_path, nullptr);
    if (storage_path_ptr == nullptr) return JNI_ERR;
    const std::string storage_file_path(storage_path_ptr);
    env->ReleaseStringUTFChars(storage_path, storage_path_ptr);

    const char *opencl_cache_full_path_ptr =
            env->GetStringUTFChars(opencl_cache_full_path, nullptr);
    if (opencl_cache_full_path_ptr == nullptr) return JNI_ERR;
    const std::string str_opencl_cache_full_path(opencl_cache_full_path_ptr);
    env->ReleaseStringUTFChars(opencl_cache_full_path,
                               opencl_cache_full_path_ptr);

    // SetStoragePath will be replaced by SetOpenCLCacheFullPath in the future
    mace_context.gpu_context = mace::GPUContextBuilder()
            .SetStoragePath(storage_file_path)
            .SetOpenCLCacheFullPath(str_opencl_cache_full_path)
            .SetOpenCLCacheReusePolicy(
                    static_cast<mace::OpenCLCacheReusePolicy>(opencl_cache_reuse_policy))
            .Finalize();

    return JNI_OK;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateGPUContextThread(
        JNIEnv *env, jclass thisObj, jstring storage_path,
        jstring opencl_cache_full_path, jint opencl_cache_reuse_policy) {
    MaceContext_thread &mace_context_thread = GetMaceContextThread();
    // DO NOT USE tmp directory.
    // Please use APP's own directory and make sure the directory exists.
    const char *storage_path_ptr = env->GetStringUTFChars(storage_path, nullptr);
    if (storage_path_ptr == nullptr) return JNI_ERR;
    const std::string storage_file_path(storage_path_ptr);
    env->ReleaseStringUTFChars(storage_path, storage_path_ptr);

    const char *opencl_cache_full_path_ptr =
            env->GetStringUTFChars(opencl_cache_full_path, nullptr);
    if (opencl_cache_full_path_ptr == nullptr) return JNI_ERR;
    const std::string str_opencl_cache_full_path(opencl_cache_full_path_ptr);
    env->ReleaseStringUTFChars(opencl_cache_full_path,
                               opencl_cache_full_path_ptr);

    // SetStoragePath will be replaced by SetOpenCLCacheFullPath in the future
    mace_context_thread.gpu_context_thread = mace::GPUContextBuilder()
            .SetStoragePath(storage_file_path)
            .SetOpenCLCacheFullPath(str_opencl_cache_full_path)
            .SetOpenCLCacheReusePolicy(
                    static_cast<mace::OpenCLCacheReusePolicy>(opencl_cache_reuse_policy))
            .Finalize();

    return JNI_OK;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateGPUContextThread3(
        JNIEnv *env, jclass thisObj, jstring storage_path,
        jstring opencl_cache_full_path, jint opencl_cache_reuse_policy) {
    MaceContextThread3 &mace_context_thread3 = GetMaceContextThread3();
    // DO NOT USE tmp directory.
    // Please use APP's own directory and make sure the directory exists.
    const char *storage_path_ptr = env->GetStringUTFChars(storage_path, nullptr);
    if (storage_path_ptr == nullptr) return JNI_ERR;
    const std::string storage_file_path(storage_path_ptr);
    env->ReleaseStringUTFChars(storage_path, storage_path_ptr);

    const char *opencl_cache_full_path_ptr =
            env->GetStringUTFChars(opencl_cache_full_path, nullptr);
    if (opencl_cache_full_path_ptr == nullptr) return JNI_ERR;
    const std::string str_opencl_cache_full_path(opencl_cache_full_path_ptr);
    env->ReleaseStringUTFChars(opencl_cache_full_path,
                               opencl_cache_full_path_ptr);

    // SetStoragePath will be replaced by SetOpenCLCacheFullPath in the future
    mace_context_thread3.gpu_context_thread = mace::GPUContextBuilder()
            .SetStoragePath(storage_file_path)
            .SetOpenCLCacheFullPath(str_opencl_cache_full_path)
            .SetOpenCLCacheReusePolicy(
                    static_cast<mace::OpenCLCacheReusePolicy>(opencl_cache_reuse_policy))
            .Finalize();

    return JNI_OK;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateEngine(
        JNIEnv *env, jclass thisObj, jint num_threads, jint cpu_affinity_policy,
        jint gpu_perf_hint, jint gpu_priority_hint,
        jstring model_name_str, jstring device) {
    MaceContext &mace_context = GetMaceContext();

    // get device
    const char *device_ptr = env->GetStringUTFChars(device, nullptr);
    if (device_ptr == nullptr) return JNI_ERR;
    mace_context.runtime_type = ParseDeviceType(device_ptr);
    env->ReleaseStringUTFChars(device, device_ptr);

    // create MaceEngineConfig
    mace::MaceStatus status;
    mace::MaceEngineConfig config;
    config.SetRuntimeType(mace_context.runtime_type);
    status = config.SetCPUThreadPolicy(
            num_threads,
            static_cast<mace::CPUAffinityPolicy>(cpu_affinity_policy));
    if (status != mace::MaceStatus::MACE_SUCCESS) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 1 - GPU -- image_classify attrs",
                            "threads: %d, cpu: %d",
                            num_threads, cpu_affinity_policy);
    }
    if (mace_context.runtime_type == mace::RuntimeType::RT_OPENCL) {
        config.SetGPUContext(mace_context.gpu_context);
        config.SetGPUHints(
                static_cast<mace::GPUPerfHint>(gpu_perf_hint),
                static_cast<mace::GPUPriorityHint>(gpu_priority_hint));
        __android_log_print(ANDROID_LOG_INFO,
                            "Thread 1 - GPU -- image_classify attrs",
                            "gpu perf: %d, priority: %d",
                            gpu_perf_hint, gpu_priority_hint);
    }

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 1 - GPU -- image_classify attrs",
                        "device: %d",
                        mace_context.runtime_type);

    //  parse model name
    const char *model_name_ptr = env->GetStringUTFChars(model_name_str, nullptr);
    if (model_name_ptr == nullptr) return JNI_ERR;
    mace_context.model_name.assign(model_name_ptr);
    env->ReleaseStringUTFChars(model_name_str, model_name_ptr);

    //  load model input and output name
    auto model_info_iter =
            mace_context.model_infos.find(mace_context.model_name);
    if (model_info_iter == mace_context.model_infos.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 1 - GPU -- image_classify",
                            "Invalid model name: %s",
                            mace_context.model_name.c_str());
        return JNI_ERR;
    }
    std::vector<std::string> input_names = {model_info_iter->second.input_name};
    std::vector<std::string> output_names = {model_info_iter->second.output_name};

    mace::MaceStatus create_engine_status =
            CreateMaceEngineFromCode(mace_context.model_name,
                                     nullptr,
                                     0,
                                     input_names,
                                     output_names,
                                     config,
                                     &mace_context.engine,
                                     nullptr,
                                     nullptr,
                                     false);

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 1 - GPU -- image_classify attrs",
                        "create result: %s",
                        create_engine_status.information().c_str());

    return create_engine_status == mace::MaceStatus::MACE_SUCCESS ?
           JNI_OK : JNI_ERR;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateEngineThread(
        JNIEnv *env, jclass thisObj, jint num_threads, jint cpu_affinity_policy,
        jint gpu_perf_hint, jint gpu_priority_hint,
        jstring model_name_str, jstring device) {
    LOGI("In maceMobilenetCreateEngineThread first line !!!!!!!");

    MaceContext_thread &mace_context_thread = GetMaceContextThread();

    // get device
    const char *device_ptr = env->GetStringUTFChars(device, nullptr);
    if (device_ptr == nullptr) return JNI_ERR;
    mace_context_thread.runtime_type_thread = ParseDeviceType(device_ptr);
    env->ReleaseStringUTFChars(device, device_ptr);

    // create MaceEngineConfig
    mace::MaceStatus status;
    mace::MaceEngineConfig config;
    config.SetRuntimeType(mace_context_thread.runtime_type_thread);
    status = config.SetCPUThreadPolicy(
            num_threads,
            static_cast<mace::CPUAffinityPolicy>(cpu_affinity_policy));
    if (status != mace::MaceStatus::MACE_SUCCESS) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 2 - CPU -- image_classify attrs",
                            "threads: %d, cpu: %d",
                            num_threads, cpu_affinity_policy);
    }
    if (mace_context_thread.runtime_type_thread == mace::RuntimeType::RT_OPENCL) {
        config.SetGPUContext(mace_context_thread.gpu_context_thread);
        config.SetGPUHints(
                static_cast<mace::GPUPerfHint>(gpu_perf_hint),
                static_cast<mace::GPUPriorityHint>(gpu_priority_hint));
        __android_log_print(ANDROID_LOG_INFO,
                            "Thread 2 - CPU -- image_classify attrs",
                            "gpu perf: %d, priority: %d",
                            gpu_perf_hint, gpu_priority_hint);
    }

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 2 - CPU -- image_classify attrs",
                        "device: %d",
                        mace_context_thread.runtime_type_thread);

    //  parse model name
    const char *model_name_ptr = env->GetStringUTFChars(model_name_str, nullptr);
    if (model_name_ptr == nullptr) return JNI_ERR;
    mace_context_thread.model_name_thread.assign(model_name_ptr);
    env->ReleaseStringUTFChars(model_name_str, model_name_ptr);

    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    //  load model input and output name
    auto model_info_iter =
            mace_context_thread.model_infos_thread.find(mace_context_thread.model_name_thread);
    if (model_info_iter == mace_context_thread.model_infos_thread.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 2 - CPU -- image_classify",
                            "Invalid model name: %s",
                            mace_context_thread.model_name_thread.c_str());
        return JNI_ERR;
    }
    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    std::vector<std::string> input_names = {model_info_iter->second.input_name};
    std::vector<std::string> output_names = {model_info_iter->second.output_name};

    mace::MaceStatus create_engine_status =
            CreateMaceEngineFromCode(mace_context_thread.model_name_thread,
                                     nullptr,
                                     0,
                                     input_names,
                                     output_names,
                                     config,
                                     &mace_context_thread.engine_thread,
                                     nullptr,
                                     nullptr,
                                     false);
    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 2 - CPU -- image_classify attrs",
                        "create result: %s",
                        create_engine_status.information().c_str());

    LOGI("In maceMobilenetCreateEngineThread final line !!!!!!!");
    return create_engine_status == mace::MaceStatus::MACE_SUCCESS ?
    JNI_OK : JNI_ERR;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetCreateEngineThread3(
        JNIEnv *env, jclass thisObj, jint num_threads, jint cpu_affinity_policy,
        jint gpu_perf_hint, jint gpu_priority_hint,
        jstring model_name_str, jstring device) {
    LOGI("In maceMobilenetCreateEngineThread first line !!!!!!!");

    MaceContextThread3 &mace_context_thread3 = GetMaceContextThread3();

    // get device
    const char *device_ptr = env->GetStringUTFChars(device, nullptr);
    if (device_ptr == nullptr) return JNI_ERR;
    mace_context_thread3.runtime_type_thread = ParseDeviceType(device_ptr);
    env->ReleaseStringUTFChars(device, device_ptr);

    // create MaceEngineConfig
    mace::MaceStatus status;
    mace::MaceEngineConfig config;
    config.SetRuntimeType(mace_context_thread3.runtime_type_thread);
    status = config.SetCPUThreadPolicy(
            num_threads,
            static_cast<mace::CPUAffinityPolicy>(cpu_affinity_policy));
    if (status != mace::MaceStatus::MACE_SUCCESS) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 2 - CPU -- image_classify attrs",
                            "threads: %d, cpu: %d",
                            num_threads, cpu_affinity_policy);
    }
    if (mace_context_thread3.runtime_type_thread == mace::RuntimeType::RT_OPENCL) {
        config.SetGPUContext(mace_context_thread3.gpu_context_thread);
        config.SetGPUHints(
                static_cast<mace::GPUPerfHint>(gpu_perf_hint),
                static_cast<mace::GPUPriorityHint>(gpu_priority_hint));
        __android_log_print(ANDROID_LOG_INFO,
                            "Thread 2 - CPU -- image_classify attrs",
                            "gpu perf: %d, priority: %d",
                            gpu_perf_hint, gpu_priority_hint);
    }

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 2 - CPU -- image_classify attrs",
                        "device: %d",
                        mace_context_thread3.runtime_type_thread);

    //  parse model name
    const char *model_name_ptr = env->GetStringUTFChars(model_name_str, nullptr);
    if (model_name_ptr == nullptr) return JNI_ERR;
    mace_context_thread3.model_name_thread.assign(model_name_ptr);
    env->ReleaseStringUTFChars(model_name_str, model_name_ptr);

    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    //  load model input and output name
    auto model_info_iter =
            mace_context_thread3.model_infos_thread.find(mace_context_thread3.model_name_thread);
    if (model_info_iter == mace_context_thread3.model_infos_thread.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "Thread 2 - CPU -- image_classify",
                            "Invalid model name: %s",
                            mace_context_thread3.model_name_thread.c_str());
        return JNI_ERR;
    }
    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    std::vector<std::string> input_names = {model_info_iter->second.input_name};
    std::vector<std::string> output_names = {model_info_iter->second.output_name};

    mace::MaceStatus create_engine_status =
            CreateMaceEngineFromCode(mace_context_thread3.model_name_thread,
                                     nullptr,
                                     0,
                                     input_names,
                                     output_names,
                                     config,
                                     &mace_context_thread3.engine_thread,
                                     nullptr,
                                     nullptr,
                                     false);
    LOGI("In maceMobilenetCreateEngineThread mid line !!!!!!!");

    __android_log_print(ANDROID_LOG_INFO,
                        "Thread 2 - CPU -- image_classify attrs",
                        "create result: %s",
                        create_engine_status.information().c_str());

    LOGI("In maceMobilenetCreateEngineThread final line !!!!!!!");
    return create_engine_status == mace::MaceStatus::MACE_SUCCESS ?
           JNI_OK : JNI_ERR;
}


void GPUInfer(MaceContext &mace_context, std::map<std::string,
              mace::MaceTensor> &inputs, std::map<std::string, mace::MaceTensor> &outputs)
{
    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    mace_context.engine->Run(inputs, &outputs);

    auto finish = std::chrono::high_resolution_clock::now();
    // calculate the time diff and put it in int
    int time_diff = std::chrono::duration_cast<milli>(finish - start).count();
    cumulative_time_GPU = cumulative_time_GPU + time_diff;

    LOGI("In frame no %d MACE Thread1 GPU model took %d milliseconds. Cumulative time is %d",
         frame_no, time_diff, cumulative_time_GPU);
}

void CPUInfer(MaceContext_thread &mace_context_thread, std::map<std::string,
        mace::MaceTensor> &inputs_thread, std::map<std::string, mace::MaceTensor> &outputs_thread)
{
    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    mace_context_thread.engine_thread->Run(inputs_thread, &outputs_thread);

    auto finish = std::chrono::high_resolution_clock::now();
    // calculate the time diff and put it in int
    int time_diff = std::chrono::duration_cast<milli>(finish - start).count();
    cumulative_time_CPU = cumulative_time_CPU + time_diff;

    LOGI("In frame no %d MACE Thread2 CPU model took %d milliseconds. Cumulative time is %d",
         frame_no, time_diff, cumulative_time_CPU);

}

void Thread3Infer(MaceContextThread3 &mace_context_thread, std::map<std::string,
        mace::MaceTensor> &inputs_thread, std::map<std::string, mace::MaceTensor> &outputs_thread)
{
    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    mace_context_thread.engine_thread->Run(inputs_thread, &outputs_thread);

    auto finish = std::chrono::high_resolution_clock::now();
    // calculate the time diff and put it in int
    int time_diff = std::chrono::duration_cast<milli>(finish - start).count();
    cumulative_time_thread3 = cumulative_time_thread3 + time_diff;

    LOGI("In frame no %d MACE Thread3 CPU model took %d milliseconds. Cumulative time is %d",
         frame_no, time_diff, cumulative_time_thread3);

}

JNIEXPORT jfloatArray JNICALL
Java_com_xiaomi_mace_JniMaceUtils_maceMobilenetClassify(
        JNIEnv *env, jclass thisObj, jfloatArray input_data) {
//    LOGI("In maceMobilenetClassify first line !!!!!!!!!!!!!");

    MaceContext &mace_context = GetMaceContext();
    MaceContext_thread &mace_context_thread = GetMaceContextThread();
    MaceContextThread3 &mace_context_thread3 = GetMaceContextThread3();

    //  prepare input and output
    auto model_info_iter =
            mace_context.model_infos.find(mace_context.model_name);
    if (model_info_iter == mace_context.model_infos.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "image_classify",
                            "Invalid model name: %s",
                            mace_context.model_name.c_str());
        return nullptr;
    }

    auto model_info_iter_thread =
            mace_context_thread.model_infos_thread.find(mace_context_thread.model_name_thread);
    if (model_info_iter_thread == mace_context_thread.model_infos_thread.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "CPU ---- image_classify",
                            "Invalid model name: %s",
                            mace_context_thread.model_name_thread.c_str());
        return nullptr;
    }

    auto model_info_iter_thread3 =
            mace_context_thread3.model_infos_thread.find(mace_context_thread3.model_name_thread);
    if (model_info_iter_thread3 == mace_context_thread3.model_infos_thread.end()) {
        __android_log_print(ANDROID_LOG_ERROR,
                            "CPU ---- image_classify",
                            "Invalid model name: %s",
                            mace_context_thread3.model_name_thread.c_str());
        return nullptr;
    }

    const ModelInfo &model_info = model_info_iter->second;
    const ModelInfo &model_info_thread = model_info_iter_thread->second;
    const ModelInfo &model_info_thread3 = model_info_iter_thread3->second;

    const std::string &input_name = model_info.input_name;
    const std::string &output_name = model_info.output_name;
    const std::vector<int64_t> &input_shape = model_info.input_shape;
    const std::vector<int64_t> &output_shape = model_info.output_shape;
    const int64_t input_size =
            std::accumulate(input_shape.begin(), input_shape.end(), 1,
                            std::multiplies<int64_t>());
    const int64_t output_size =
            std::accumulate(output_shape.begin(), output_shape.end(), 1,
                            std::multiplies<int64_t>());

    const std::string &input_name_thread = model_info_thread.input_name;
    const std::string &output_name_thread = model_info_thread.output_name;
    const std::vector<int64_t> &input_shape_thread = model_info_thread.input_shape;
    const std::vector<int64_t> &output_shape_thread = model_info_thread.output_shape;
    const int64_t input_size_thread = std::accumulate(input_shape_thread.begin(), input_shape_thread.end(), 1,
                            std::multiplies<int64_t>());
    const int64_t output_size_thread = std::accumulate(output_shape_thread.begin(), output_shape_thread.end(), 1,
                            std::multiplies<int64_t>());

    const std::string &input_name_thread3 = model_info_thread3.input_name;
    const std::string &output_name_thread3 = model_info_thread3.output_name;
    const std::vector<int64_t> &input_shape_thread3 = model_info_thread3.input_shape;
    const std::vector<int64_t> &output_shape_thread3 = model_info_thread3.output_shape;
    const int64_t input_size_thread3 = std::accumulate(input_shape_thread3.begin(), input_shape_thread3.end(), 1,
                                                      std::multiplies<int64_t>());
    const int64_t output_size_thread3 = std::accumulate(output_shape_thread3.begin(), output_shape_thread3.end(), 1,
                                                       std::multiplies<int64_t>());

    //  load input
    jfloat *input_data_ptr = env->GetFloatArrayElements(input_data, nullptr);

    if (input_data_ptr == nullptr) return nullptr;
    jsize length = env->GetArrayLength(input_data);
//    if (length != input_size) return nullptr; // commented

    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;

    std::map<std::string, mace::MaceTensor> inputs_thread;
    std::map<std::string, mace::MaceTensor> outputs_thread;

    std::map<std::string, mace::MaceTensor> inputs_thread3;
    std::map<std::string, mace::MaceTensor> outputs_thread3;

    // construct input
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    // fill data in buffer_in
//    std::copy_n(input_data_ptr, input_size, buffer_in.get());

    // start
    for (int j = 0; j < input_size; ++j)
    {   buffer_in.get()[j] = 128.f; //0.5
    }
    // end
    env->ReleaseFloatArrayElements(input_data, input_data_ptr, 0);

    inputs[input_name] = mace::MaceTensor(input_shape, buffer_in,
                                          mace::DataFormat::NHWC);

    // construct input // madhu
    auto buffer_in_thread = std::shared_ptr<float>(new float[input_size_thread],
                                            std::default_delete<float[]>());
    for (int j = 0; j < input_size_thread; ++j)
    {   buffer_in_thread.get()[j] = 128.f; //0.5
    }
    inputs_thread[input_name_thread] = mace::MaceTensor(input_shape_thread, buffer_in_thread,
                                          mace::DataFormat::NHWC);

    auto buffer_in_thread3 = std::shared_ptr<float>(new float[input_size_thread3],
                                                   std::default_delete<float[]>());
    for (int j = 0; j < input_size_thread3; ++j)
    {   buffer_in_thread3.get()[j] = 128.f; //0.5
    }
    inputs_thread3[input_name_thread3] = mace::MaceTensor(input_shape_thread3, buffer_in_thread3,
                                                        mace::DataFormat::NHWC);
    // end

    // construct output
    auto buffer_out = std::shared_ptr<float>(new float[output_size_thread],
                                             std::default_delete<float[]>());
    outputs[output_name] = mace::MaceTensor(output_shape, buffer_out,
                                            mace::DataFormat::NHWC);

    auto buffer_out_thread = std::shared_ptr<float>(new float[output_size_thread],
                                             std::default_delete<float[]>());
    outputs_thread[output_name_thread] = mace::MaceTensor(output_shape_thread, buffer_out_thread,
                                            mace::DataFormat::NHWC);

    auto buffer_out_thread3 = std::shared_ptr<float>(new float[output_size_thread3],
                                                    std::default_delete<float[]>());
    outputs_thread3[output_name_thread3] = mace::MaceTensor(output_shape_thread3, buffer_out_thread3,
                                                          mace::DataFormat::NHWC);

    using milli = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();

    std::thread GPUThread (GPUInfer, std::ref(mace_context), std::ref(inputs), std::ref(outputs));  // spawn new thread that calls CPUThread
    std::thread CPUThread (CPUInfer, std::ref(mace_context_thread), std::ref(inputs_thread), std::ref(outputs_thread));  // spawn new thread that calls GPUThread
    std::thread Thread3 (Thread3Infer, std::ref(mace_context_thread3), std::ref(inputs_thread3), std::ref(outputs_thread3));  // spawn new thread that calls GPUThread

    LOGI("main, CPUThread and GPUThread now execute concurrently...");

    // synchronize threads:
    GPUThread.join();                // pauses until first finishes
    CPUThread.join();                // pauses until second finishes
    Thread3.join();                // pauses until third finishes

    LOGI("CPUThread and GPUThread completed inference");


//    // run model
//    mace_context.engine->Run(inputs, &outputs);
//
    auto finish = std::chrono::high_resolution_clock::now();
    // calculate the time diff and put it in int
    int time_diff = std::chrono::duration_cast<milli>(finish - start).count();
    cumulative_time = cumulative_time + time_diff;

    LOGI("In frame no %d MACE two threads combined took %d milliseconds. Cumulative time is %d",
         frame_no, time_diff, cumulative_time);
    frame_no++; // = frame_no + 1;

    if (mace_context.runtime_type != mace::RuntimeType::RT_OPENCL) {
        LOGI("First model not running on GPU"); }
//    }else{
//        LOGI("First model running on CPU");
//    }
    if (mace_context_thread.runtime_type_thread == mace::RuntimeType::RT_OPENCL) {
        LOGI("Second model not running on CPU"); }
//    }else{
//        LOGI("Second model running on CPU");
//    }
    if (mace_context_thread3.runtime_type_thread == mace::RuntimeType::RT_OPENCL) {
        LOGI("Third model not running on CPU"); }
    // transform output
    jfloatArray jOutputData = env->NewFloatArray(output_size);  // allocate
    jfloatArray jOutputData_thread = env->NewFloatArray(output_size_thread);  // allocate

    if (jOutputData == nullptr) return nullptr;
    env->SetFloatArrayRegion(jOutputData, 0, output_size,
                             outputs[output_name].data().get());  // copy

    return jOutputData;
}

