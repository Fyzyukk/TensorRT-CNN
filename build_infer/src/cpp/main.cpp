# include "model.hpp"
# include "utils.hpp"
#include "iostream"
using namespace std;
int main() {
//    cout<<"111111"<<endl;
    // 命名格式: 直接以文件夹开头, 不要加 ./
    Model model("D:/work/TensorRT-main/build_infer/model/resnet50.onnx", Model::precision::FP32);

    if(!model.build()){
        LOGE("fail in building model");
        return 0;
    }
#if 1
    if(!model.infer("D:/work/TensorRT-main/build_infer/data/fox.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("D:/work/TensorRT-main/build_infer/data/cat.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("D:/work/TensorRT-main/build_infer/data/eagle.png")){
        LOGE("fail in infering model");
        return 0;
    }

    if(!model.infer("D:/work/TensorRT-main/build_infer/data/gazelle.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("D:/work/TensorRT-main/build_infer/data/tiny-cat.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("D:/work/TensorRT-main/build_infer/data/unknown.png")){
        LOGE("fail in infering model");
        return 0;
    }
    if(!model.infer("D:/work/TensorRT-main/build_infer/data/wolf.png")){
        LOGE("fail in infering model");
        return 0;
    }
#endif
    return 0;
}