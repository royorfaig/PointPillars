import argparse
import torch
import torch.onnx
from model import PointPillars
import cv2
import numpy as np
import os
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as gs
def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError

#Function to Convert to ONNX
def Convert_ONNX(args):

    # CLASSES = {
    #     'Trunk': 0
    # }
    # model = PointPillars(nclasses=len(CLASSES)).cuda()
    # model.load_state_dict(torch.load(args.ckpt))

    CLASSES = {
        'Trunk': 0
    }
    #model = PointPillars(nclasses=len(CLASSES)).cuda()
    # state_dict = torch.load('/home/roy/Projects/trunk_detector/pretrained/epoch_100.pth')
    pth='/home/roy/Projects/trunk_detector/pretrained/epoch_100.pth'
    model = PointPillars(nclasses=len(CLASSES)).cuda()
    #model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
     model.load_state_dict(torch.load(pth))
    # model = PointPillars(nclasses=len(CLASSES)).cuda()
    # model.eval()
    #model = Network()
    path = "myFirstModel.pth"
    path ='/home/roy/Projects/trunk_detector/pretrained/epoch_100.pth'
    #points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
    #model.load_state_dict(torch.load(path))

    file_name = args.pc_path
    pc = read_points(file_name)
    pc_torch = torch.from_numpy(pc)
    pc_torch = pc_torch.cuda()
    wp=torch.load(path)
    #torch.onnx.export(model, [pc_torch], "PointPillars100epocs.onnx",verbose=False,input_names = ['modelInput'],output_names = ['modelOutput'],export_params=True)
    #torch.onnx.export(model, [pc_torch], "PointPillars100epocs.onnx")
    # Export the model

    dummy_input = dict()
    dummy_input['batched_pts']=[pc_torch]
    dummy_input['voxel_size'] = [0.16, 0.16, 4]
    dummy_input['point_cloud_range'] = [0, -39.68, -3, 69.12, 39.68, 1]
    dummy_input['max_num_points'] = 64
    dummy_input['max_voxels'] = (16000, 40000)

    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "PointPillars_100epocs.onnx",       # where to save the model
         opset_version=14,
         do_constant_folding=True,  # whether to execute constant folding for optimization
         keep_initializers_as_inputs=True,
         export_params=True,  # store the trained parameter weights inside the model file
         input_names = ['voxel_size', 'point_cloud_range', 'max_num_points','max_voxels'],   # the model's input names
         output_names =  ['lidar_bboxes', 'labels', 'scores'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    #do_constant_folding = True,  # whether to execute constant folding for optimization
    print(" ")
    print('Model has been converted to ONNX')
   #opset_version=10,    # the ONNX version to export the model to

    #fpath="/home/roy/Downloads/pfe.onnx"
    onnx_model = onnx.load("./PointPillars100epocs.onnx")
    #print(f"The model is:\n{onnx_model}")

    # Check the model
    try:
        resonnx=onnx.checker.check_model(onnx_model)
        print("Res {}.",resonnx)
    except onnx.checker.ValidationError as e:
        print("The model is invalid:")
    else:
        print("The model is valid!")
    # graph = gs.import_onnx(onnx_model)
    # tmap = graph.tensors()
    # tmp_inputs = graph.inputs
    # print(tmp_inputs)
    # model2 = onnx.load()

    # Check that the model is well formed
   # onnx.checker.check_model(model2)

    # Print a human readable representation of the graph
    #print(onnx.helper.printable_graph(model2.graph))

    # Below is for optimizing performance
   # sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = 24
    # ...
    #ort_session = ort.InferenceSession() #, sess_options=sess_options)
    #session = ort.InferenceSession("./PointPillars100epocs.onnx", None, providers=["CUDAExecutionProvider"])
    session = ort.InferenceSession("./PointPillars100epocs.onnx", providers=["CUDAExecutionProvider"])


    # Load and preprocess the input image inputTensor
    ...

    # Run inference
    #session = ort.InferenceSession("PointPillars100epocs.onnx")
    # outputs = session.run(None, {"input": inputTensor})
    # print(outputs)

if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='/home/roy/Projects/trunk_detector/pretrained/epoch_100.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', default='/home/roy/Projects/trunk_detector/tests/008024_52804.bin', help='your point cloud path')
    # parser.add_argument('--no_cuda', action='store_true',
    #                     help='whether to use cuda')
    parser.add_argument('--res_path', default='/home/roy/Projects/trunk_detector/html_res', help='your output html path')

    args = parser.parse_args()



    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX(args)

    # import onnxruntime as ort
    #
    # # Load the model and create InferenceSession
    # model_path = "path/to/your/onnx/model"
    # session = ort.InferenceSession(model_path)
    #
    # # Load and preprocess the input image inputTensor
    # ...
    #
    # # Run inference
    # outputs = session.run(None, {"input": inputTensor})
    # print(outputs)