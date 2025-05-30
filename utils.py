from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib as mpl
import torch.nn.functional as F
from torchvision import datasets, transforms
#from skimage.feature import hog
from scipy.ndimage import gaussian_filter

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_size(path):
    image = Image.open(path).convert('L')
    w,h=image.size
    return w,h


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = Image.open(path).convert('L')
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    elif mode == 'YCbCr':
        img = Image.open(path).convert('YCbCr')
        image, _, _ = img.split()

    if height is not None and width is not None:
        image = image.resize((width, height), resample=Image.NEAREST)

    image = np.array(image)

    return image


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode) / 255.0
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
    
def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode) / 255.0

        if mode == 'L' :#or mode == 'YCbCr':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()

        
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    image = Image.fromarray(data)
    image = image.convert('L')
    image.save(path)


def PixelIntensityDecision(latlrr_image,ir_image,vi_image):
    mask = torch.where(latlrr_image > 30, 1, 0)
    vi_mask = vi_image * mask
    ir_mask = ir_image * mask
    max_input_pixel_mask = torch.max(vi_mask, ir_mask)
    max_input_pixel = vi_image - vi_mask + max_input_pixel_mask
    return max_input_pixel,mask



def gradient_loss(predicted, target1, target2):
    gradient_predicted_x = torch.abs(F.conv2d(predicted, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_predicted_y = torch.abs(F.conv2d(predicted, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    gradient_target1_x = torch.abs(F.conv2d(target1, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_target1_y = torch.abs(F.conv2d(target1, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    gradient_target2_x = torch.abs(F.conv2d(target2, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_target2_y = torch.abs(F.conv2d(target2, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    loss = F.l1_loss(gradient_predicted_x, torch.max(gradient_target1_x,gradient_target2_x)) + F.l1_loss(gradient_predicted_y, torch.max(gradient_target1_y,gradient_target2_y))

    return loss



def median_filter_torch(image_tensor, kernel_size):

    image_np = image_tensor.cpu().squeeze(0).squeeze(0).numpy()

    image_np_uint8 = (image_np * 255).astype(np.uint8)

    img_height, img_width = image_np_uint8.shape
    pad = kernel_size // 2
    filtered_image = np.zeros_like(image_np_uint8)

    for i in range(pad, img_height - pad):
        for j in range(pad, img_width - pad):
            window = image_np_uint8[i - pad:i + pad + 1, j - pad:j + pad + 1]
            median_val = np.median(window)
            filtered_image[i, j] = median_val

    filtered_tensor = torch.from_numpy(filtered_image / 255.0).float().unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    return filtered_tensor






