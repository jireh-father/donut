import torch
import argparse
import torch.nn as nn
import time
import os

model_classifier_map = {
    'alexnet': ['classifier', 6],
    'vgg': ['classifier', 6],
    'mobilenet_v3': ['classifier', 3],
    'mobilenet': ['classifier', 1],
    'mnasnet': ['classifier', 6],
    'resnet': ['fc'],
    'inception': ['fc'],
    'googlenet': ['fc'],
    'shufflenet': ['fc'],
    'densenet': ['classifier'],
    'resnext': ['fc'],
    'wide_resnet': ['fc'],
    'efficientnet': ['_fc'],
    'bagnet': ['fc'],
    'rexnet': ['output', 1],
}


def init_model(model_name, num_classes):
    if model_name.startswith("efficientnet-lite"):
        from efficientnet_lite_pytorch import EfficientNet
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
        return model

    if model_name.startswith("efficientnet"):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        return model

    from torchvision import models
    for m_key in model_classifier_map:
        if m_key in model_name:
            model_fn = getattr(models, model_name)
            cls_layers = model_classifier_map[m_key]

            if model_name.startswith("inception"):
                # input_size = 299
                model = model_fn(aux_logits=False)
            else:
                # input_size = 224
                model = model_fn()

            if len(cls_layers) == 1:
                in_features = getattr(model, cls_layers[0]).in_features
                setattr(model, cls_layers[0], nn.Linear(in_features, num_classes))
            else:
                print(cls_layers)
                classifier = getattr(model, cls_layers[0])
                in_features = classifier[cls_layers[1]].in_features
                classifier[cls_layers[1]] = nn.Linear(in_features, num_classes)
            return model


def main(args):
    model = init_model(args.model_name, args.num_classes)
    if args.model_name.startswith("efficientnet"):
        model.set_swish(memory_efficient=False)

    device = 'cuda' if args.use_cuda else 'cpu'

    checkpoint_dict = torch.load(args.model_path, map_location=device if not args.use_cuda else None)
    pretrained_dict = checkpoint_dict['state_dict']

    try:
        model.load_state_dict(pretrained_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(pretrained_dict)
        model = model.module

    model = model.to(device)
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, args.input_size, args.input_size, requires_grad=True).to(device)

    # 모델 변환
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.onnx.export(model,  # 실행될 모델
                      x,  # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                      args.output_path,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                      export_params=True,  # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                      opset_version=args.opset_version,  # 모델을 변환할 때 사용할 ONNX 버전
                      do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                      input_names=['input'],  # 모델의 입력값을 가리키는 이름
                      output_names=['output'],  # 모델의 출력값을 가리키는 이름
                      dynamic_axes={'input': {2: 'height', 3: 'width'}} if args.use_dynamic_axes else {}
                      )
    print("finished to convert pytorch model to onnx model.", args.output_path)

    if args.test:
        print("started to compare two models")
        torch_out = model(x)
        test_cnt = 10
        pytorch_total_time = 0.
        for i in range(test_cnt):
            start = time.time()
            model(x)
            pytorch_total_time += time.time() - start

        import onnxruntime
        ort_session = onnxruntime.InferenceSession(args.output_path)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        # ort_outs = ort_session.run(None, ort_inputs)

        onnx_total_time = 0.
        for i in range(test_cnt):
            start = time.time()
            ort_session.run(None, ort_inputs)
            onnx_total_time += time.time() - start

        print("onnx model inference time", onnx_total_time / test_cnt)
        print("pytorch model inference time", pytorch_total_time / test_cnt)
        import numpy as np
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--num_classes', default=None, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--opset_version', default=9, type=int)

    parser.add_argument('--use_dynamic_axes', default=False, action="store_true")
    parser.add_argument('--test', default=False, action="store_true")
    parser.add_argument('--use_cuda', default=False, action="store_true")
    main(parser.parse_args())

    print("done")
