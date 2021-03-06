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

package com.xiaomi.mace.demo;

import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import com.xiaomi.mace.JniMaceUtils;
import com.xiaomi.mace.demo.camera.MessageEvent;
import com.xiaomi.mace.demo.result.InitData;
import com.xiaomi.mace.demo.result.LabelCache;
import com.xiaomi.mace.demo.result.ResultData;

import org.greenrobot.eventbus.EventBus;

import java.util.Arrays;

public class AppModel {

    private boolean stopClassify = false;
    private Handler mJniThread;
    public static AppModel instance = new AppModel();

    private AppModel() {
        Log.d("myTag", "This is in AppModel");
        HandlerThread thread = new HandlerThread("jniThread");
        thread.start();
        mJniThread = new Handler(thread.getLooper());
    }

    public void maceMobilenetCreateGPUContext(final InitData initData) {
        mJniThread.post(new Runnable() {
            @Override
            public void run() {
                Log.d("myTag", "This is in maceMobilenetCreateGPUContext run()");
                int result = JniMaceUtils.maceMobilenetCreateGPUContext(
                        initData.getStoragePath(),
                        initData.getOpenclCacheFullPath(),
                        initData.getOpenclCacheReusePolicy());
                Log.i("APPModel", "maceMobilenetCreateGPUContext result = " + result);
            }
        });
    }

    public void maceMobilenetCreateGPUContextThread(final InitData initData) {
        mJniThread.post(new Runnable() {
            @Override
            public void run() {
                Log.d("myTag", "This is in maceMobilenetCreateGPUContextThread run()");
                int result = JniMaceUtils.maceMobilenetCreateGPUContextThread(
                        initData.getStoragePath_GPU(),
                        initData.getOpenclCacheFullPath_GPU(),
                        initData.getOpenclCacheReusePolicy());
                Log.i("APPModel", "maceMobilenetCreateGPUContextGPUThread result = " + result);
            }
        });
    }

    public void maceMobilenetCreateEngine(final InitData initData, final CreateEngineCallback callback) {
//    public void maceMobilenetCreateEngineThread(final InitData initData, final CreateEngineCallback callback) {
        mJniThread.post(new Runnable() {
            @Override
            public void run() {
                Log.d("myTag", "This is in maceMobilenetCreateEngine run()");
                int result = JniMaceUtils.maceMobilenetCreateEngine(
                        initData.getOmpNumThreads(), initData.getCpuAffinityPolicy(),
                        initData.getGpuPerfHint(), initData.getGpuPriorityHint(),
                        initData.getModel(), initData.getDevice());
                Log.i("APPModel", "maceMobilenetCreateEngine result = " + result);

                if (result == -1) {
                    stopClassify = true;
                    MaceApp.app.mMainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            callback.onCreateEngineFail(InitData.DEVICES[0].equals(initData.getDevice()));
                        }
                    });
                } else {
                    stopClassify = false;
                }
            }
        });
    }

    public void maceMobilenetCreateEngineThread(final InitData initData, final CreateEngineCallback callback) {
        mJniThread.post(new Runnable() {
            @Override
            public void run() {
                Log.d("myTag", "This is in maceMobilenetCreateEngineThread run()");
                int result = JniMaceUtils.maceMobilenetCreateEngineThread(
                        initData.getOmpNumThreads(), initData.getCpuAffinityPolicy(),
                        initData.getGpuPerfHint(), initData.getGpuPriorityHint(),
                        initData.getModel(), initData.getDevice_GPU());
                Log.i("APPModel", "maceMobilenetCreateEngineThread result = " + result);

                if (result == -1) {
                    stopClassify = true;
                    MaceApp.app.mMainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            callback.onCreateEngineFail(InitData.DEVICES[1].equals(initData.getDevice_GPU()));
                        }
                    });
                } else {
                    stopClassify = false;
                }
            }
        });
    }

    public void maceMobilenetClassify(final float[] input) {
        mJniThread.post(new Runnable() {
            @Override
            public void run() {
//                Log.d("myTag", "This is in maceMobilenetClassify run()");
                if (stopClassify) {
                    return;
                }
                long start = System.currentTimeMillis();
                float[] result = JniMaceUtils.maceMobilenetClassify(input);
//                Log.d("myTag", "Inference completed, result is : " + Arrays.toString(result));

                final ResultData resultData = LabelCache.instance().getResultFirst(result);
                resultData.costTime = System.currentTimeMillis() - start;
                EventBus.getDefault().post(new MessageEvent.MaceResultEvent(resultData));
            }
        });
    }

    public interface CreateEngineCallback {
        void onCreateEngineFail(final boolean quit);
    }

}
