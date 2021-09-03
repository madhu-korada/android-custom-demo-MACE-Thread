-> Build the app inside the mace/examples directory. If you take the app outside mace/examples you will face compilation issues. 

-> The app typically uses a `.a` library file for the model. To generate that file you need to use `build.sh` script. Checkout the modified `build_custom_models.sh` script to find out how to modify. 

-> To build the app 
```sh
cd mace/examples/android
./build.sh dynamic
# if libmace.a is needed, update `macelibrary/CMakeLists.txt` and run with `./build.sh static`.
```


To add a new model :- 
- add the model in `custom_models.yml` file
- modify the `image_classify.cc` file
- modify the `InitData.java` file

Errors :- 
```
2 files found with path 'lib/arm64-v8a/libmace.so' from inputs:
      - /home/hitech/aware/mace/examples/android-custom-demo-MACE/macelibrary/build/intermediates/merged_jni_libs/debug/out/arm64-v8a/libmace.so
      - /home/hitech/aware/mace/examples/android-custom-demo-MACE/macelibrary/build/intermediates/cxx/Debug/4j263lg2/obj/arm64-v8a/libmace.so
```
If the error is such please delete the `jniLibs` folder from the `macelibrary/src/main` folder. During the build it generates the libmace.so at 2 places, so not necessary at both places. 


-> To Install the app
---------------

```sh
# running after build step and in `mace/exampls/android` directory
adb install ./app/build/outputs/apk/app/release/app-app-release.apk
```
Or Just use the Android Studio.


-> Use arm64-v8a ABI, because our platforms support both armeabi-v7a and arm64-v8a. 



-> To Convert Keras Models
----------------

names defined in the original net won't work. open `transformer.py` file and add the below line `print("self._producer : ", self._producer)` at line 1613, in the function `def sort_by_execution(self):`. 
In the print statements you will get the full network. 
```
'activation/Softmax:0': input: "dense_1/BiasAdd_1:0"
output: "activation/Softmax:0"
name: "activation"
type: "Softmax"
arg {
  name: "T"
  i: 1
}
arg {
  name: "framework_type"
  i: 4
}
arg {
  name: "data_format"
  i: 1
}
output_shape {
  dims: 1
  dims: 10
}
```
every layer will look something like this, use `activation/Softmax:0` as output_tensor instead of the name `activation` in the yml file. 





