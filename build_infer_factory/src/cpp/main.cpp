# include <string>
# include <iostream>
# include "trt_model.hpp"
# include "trt_worker.hpp"
# include "trt_logger.hpp"


int main() 
{
    // 工厂模式: 调用即初始化
    std::string onnxPath = "D:/work/TensorRT-main/build_infer_factory/model/resnet50.onnx";

    auto level          = logger::Level::VERB;
    auto params         = model::Params();

    params.img          = {224, 224, 3};
    params.num_cls      = 1000;
    params.task         = model::task_type::CLASSIFICATION;
    params.dev          = model::device::GPU;
    params.tac          = process::tactics::GPU_BILINEAR;
    params.prec         = model::precision::INT8;
    params.cal          = model::calibrator::Entropy;

    // 创建worker实例, 在创建的时候完成初始化
    auto worker = worker::create_worker(onnxPath, level, params);

    // 推理
    # if 1
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/cat.png");
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/eagle.png");
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/fox.png");
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/gazelle.png");
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/tiny-cat.png");
    worker->inference("D:/work/TensorRT-main/build_infer_factory/data/unknown.png");
    worker->inference("D/work/TensorRT-main/build_infer_factory/data/wolf.png");
    # endif

    return 0;
}
