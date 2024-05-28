import numpy as np
import torch
import glob
import cv2
import os
import Model as Net  # Ensure this is the correct import for your model
from argparse import ArgumentParser

# Define your color palette
pallete = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
]

def relabel(img):
    label_map = {19: 255, 18: 33, 17: 32, 16: 31, 15: 28, 14: 27, 13: 26, 12: 25,
                 11: 24, 10: 23, 9: 22, 8: 21, 7: 20, 6: 19, 5: 17, 4: 13,
                 3: 12, 2: 11, 1: 8, 0: 7, 255: 0}
    for k, v in label_map.items():
        img[img == k] = v
    return img

def evaluateModel(args, model, up, image_list):
    mean = np.array([72.39, 82.90, 73.15], dtype=np.float32)
    std = np.array([45.32, 46.15, 44.91], dtype=np.float32)

    with torch.no_grad():
        for i, imgName in enumerate(image_list):
            img = cv2.imread(imgName)
            img_orig = np.copy(img) if args.overlay else None

            img = ((img - mean) / std) / 255.0
            img = cv2.resize(img, (args.inWidth, args.inHeight)).transpose((2, 0, 1))
            img_tensor = torch.from_numpy(img).unsqueeze(0).to('cuda' if args.gpu else 'cpu')
            img_out = model(img_tensor)
            img_out = up(img_out) if args.modelType == 2 else img_out

            classMap_numpy = img_out[0].max(0)[1].byte().cpu().numpy()

            if i % 100 == 0:
                print(f"Processed {i} images")

            name = os.path.basename(imgName)
            file_path_base = os.path.join(args.savedir, name.replace(args.img_extn, 'png'))

            if args.colored:
                classMap_numpy_color = np.zeros((1024, 2048, 3), dtype=np.uint8)  # Assumes output size is 512x1024
                for idx, color in enumerate(pallete):
                    classMap_numpy_color[classMap_numpy == idx] = color[::-1]  # BGR to RGB
                cv2.imwrite(file_path_base.replace('.png', '_color.png'), classMap_numpy_color)
                if args.overlay:
                    overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0)
                    cv2.imwrite(file_path_base.replace('.png', '_overlay.jpg'), overlayed)

            if args.cityFormat:
                classMap_numpy = relabel(classMap_numpy)
            cv2.imwrite(file_path_base, classMap_numpy)

def main(args):
    # Read all the images in the specified directory with the given extension
    image_list = glob.glob(os.path.join(args.data_dir, f'*.{args.img_extn}'))

    # Initialize the upsampling module if necessary
    up = None
    if args.modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
        if args.gpu:
            up = up.cuda()

    # Determine model to load based on modelType
    model_class = Net.ESPNet_Encoder if args.modelType == 2 else Net.ESPNet
    model = model_class(args.classes, args.p, args.q)
    model_subdir = 'encoder' if args.modelType == 2 else 'decoder'
    model_weight_file = os.path.join(args.weightsDir, model_subdir, f'espnet_p_{args.p}_q_{args.q}.pth')

    # Check if the weight file exists
    if not os.path.isfile(model_weight_file):
        print(f'Pre-trained model file does not exist: {model_weight_file}')
        return  # Exit if model weights not found

    # Load the model weights
    model.load_state_dict(torch.load(model_weight_file, map_location='cuda' if args.gpu else 'cpu'))

    # If using GPU, move the model to GPU
    if args.gpu:
        model = model.cuda()

    # Set model to evaluation mode
    model.eval()

    # Ensure the save directory exists
    os.makedirs(args.savedir, exist_ok=True)

    # Evaluate the model
    evaluateModel(args, model, up, image_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=2048, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=1024, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=1, help='1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=True, help='True if ESPNet. False for ESPNet-C')
    parser.add_argument('--weightsDir', default='../pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--p', default=2, type=int, help='Depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='Depth multiplier. Supported only 3, 5, 8')
    parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the segmentation masks in color')
    parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the segmentation masks overlayed on top of the RGB image')
    parser.add_argument('--classes', default=20, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    main(args)