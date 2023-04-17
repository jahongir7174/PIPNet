import copy
import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from PIL import ImageFilter


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def weight_decay(model, decay=5E-5):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.}, {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler):
    import copy
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch + 1)

    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('epoch')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


def process(data_dir, folder, image_name, label_name, target_size):
    image_path = os.path.join(data_dir, folder, image_name)
    label_path = os.path.join(data_dir, folder, label_name)

    with open(label_path, 'r') as f:
        annotation = f.readlines()[3:-1]
        annotation = [x.strip().split() for x in annotation]
        annotation = [[int(float(x[0])), int(float(x[1]))] for x in annotation]
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in annotation]
        anno_y = [x[1] for x in annotation]
        x_min = min(anno_x)
        y_min = min(anno_y)
        x_max = max(anno_x)
        y_max = max(anno_y)
        box_w = x_max - x_min
        box_h = y_max - y_min
        scale = 1.1
        x_min -= int((scale - 1) / 2 * box_w)
        y_min -= int((scale - 1) / 2 * box_h)
        box_w *= scale
        box_h *= scale
        box_w = int(box_w)
        box_h = int(box_h)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        box_w = min(box_w, image_width - x_min - 1)
        box_h = min(box_h, image_height - y_min - 1)
        annotation = [[(x - x_min) / box_w, (y - y_min) / box_h] for x, y in annotation]

        x_max = x_min + box_w
        y_max = y_min + box_h
        image_crop = image[y_min:y_max, x_min:x_max, :]
        image_crop = cv2.resize(image_crop, (target_size, target_size))
        return image_crop, annotation


def convert(data_dir, target_size=256):
    if not os.path.exists(os.path.join(data_dir, 'images', 'train')):
        os.makedirs(os.path.join(data_dir, 'images', 'train'))
    if not os.path.exists(os.path.join(data_dir, 'images', 'test')):
        os.makedirs(os.path.join(data_dir, 'images', 'test'))

    folders = ['afw', 'helen/trainset', 'lfpw/trainset']
    annotations = {}
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'train', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    annotations = {}
    folders = ['helen/testset', 'lfpw/testset', 'ibug']
    for folder in folders:
        filenames = sorted(os.listdir(os.path.join(data_dir, folder)))
        label_files = [x for x in filenames if '.pts' in x]
        image_files = [x for x in filenames if '.pts' not in x]
        assert len(image_files) == len(label_files)
        for image_name, label_name in zip(image_files, label_files):
            image_crop_name = folder.replace('/', '_') + '_' + image_name
            image_crop_name = os.path.join(data_dir, 'images', 'test', image_crop_name)

            image_crop, annotation = process(data_dir, folder, image_name, label_name, target_size)
            cv2.imwrite(image_crop_name, image_crop)
            annotations[image_crop_name] = annotation
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for image_crop_name, annotation in annotations.items():
            f.write(image_crop_name + ' ')
            for x, y in annotation:
                f.write(str(x) + ' ' + str(y) + ' ')
            f.write('\n')

    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        annotations = f.readlines()
    with open(os.path.join(data_dir, 'test_common.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' not in annotation:
                f.write(annotation)
    with open(os.path.join(data_dir, 'test_challenge.txt'), 'w') as f:
        for annotation in annotations:
            if 'ibug' in annotation:
                f.write(annotation)

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        annotations = f.readlines()
    annotations = [x.strip().split()[1:] for x in annotations]
    annotations = [[float(x) for x in anno] for anno in annotations]
    annotations = numpy.array(annotations)
    mean_face = [str(x) for x in numpy.mean(annotations, axis=0).tolist()]

    with open(os.path.join(data_dir, 'indices.txt'), 'w') as f:
        f.write(' '.join(mean_face))


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def load_weight(ckpt, model):
    dst = model.state_dict()
    src = torch.load(ckpt, 'cpu')['model'].float().state_dict()

    ckpt = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v
    model.load_state_dict(state_dict=src, strict=False)
    return model


def compute_indices(mean_face_file, params):
    with open(mean_face_file) as f:
        mean_face = f.readlines()[0]

    mean_face = mean_face.strip().split()
    mean_face = [float(x) for x in mean_face]
    mean_face = numpy.array(mean_face).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    mean_indices = []
    for i in range(mean_face.shape[0]):
        pt = mean_face[i, :]
        dists = numpy.sum(numpy.power(pt - mean_face, 2), axis=1)
        indices = numpy.argsort(dists)
        mean_indices.append(indices[1:1 + params['num_nb']])

    # each landmark predicted by X neighbors, X varies
    mean_face_indices_reversed = {}
    for i in range(mean_face.shape[0]):
        mean_face_indices_reversed[i] = [[], []]
    for i in range(mean_face.shape[0]):
        for j in range(params['num_nb']):
            mean_face_indices_reversed[mean_indices[i][j]][0].append(i)
            mean_face_indices_reversed[mean_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(mean_face.shape[0]):
        if len(mean_face_indices_reversed[i][0]) > max_len:
            max_len = len(mean_face_indices_reversed[i][0])

    # tricks, make them have equal length for efficient computation
    for i in range(mean_face.shape[0]):
        mean_face_indices_reversed[i][0] += mean_face_indices_reversed[i][0] * 10
        mean_face_indices_reversed[i][1] += mean_face_indices_reversed[i][1] * 10
        mean_face_indices_reversed[i][0] = mean_face_indices_reversed[i][0][:max_len]
        mean_face_indices_reversed[i][1] = mean_face_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(mean_face.shape[0]):
        reverse_index1 += mean_face_indices_reversed[i][0]
        reverse_index2 += mean_face_indices_reversed[i][1]
    return mean_indices, reverse_index1, reverse_index2, max_len


def compute_nme(output, target, norm):
    output = output.reshape((-1, 2))
    target = target.reshape((-1, 2))
    return numpy.mean(numpy.linalg.norm(output - target, axis=1)) / norm


def resample():
    return random.choice((Image.NEAREST, Image.BILINEAR, Image.BICUBIC))


class RandomTranslate:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            h, w = image.size
            a = 1
            b = 0
            c = int((random.random() - 0.5) * 60)
            d = 0
            e = 1
            f = int((random.random() - 0.5) * 60)
            image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample())
            label = label.copy()
            label = label.reshape(-1, 2)
            label[:, 0] -= 1. * c / w
            label[:, 1] -= 1. * f / h
            label = label.flatten()
            label[label < 0] = 0
            label[label > 1] = 1
            return image, label
        else:
            return image, label


class RandomRotate:
    def __init__(self, angle=45, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            num_lms = int(len(label) / 2)

            center_x = 0.5
            center_y = 0.5

            label = numpy.array(label) - numpy.array([center_x, center_y] * num_lms)
            label = label.reshape(num_lms, 2)
            theta = random.uniform(-numpy.radians(self.angle), +numpy.radians(self.angle))
            angle = numpy.degrees(theta)
            image = image.rotate(angle, resample=resample())

            cos = numpy.cos(theta)
            sin = numpy.sin(theta)
            label = numpy.matmul(label, numpy.array(((cos, -sin), (sin, cos))))
            label = label.reshape(num_lms * 2) + numpy.array([center_x, center_y] * num_lms)
            return image, label
        else:
            return image, label


class RandomFlip:
    def __init__(self, params, p=0.5):
        self.flip_index = (numpy.array(params['flip_index']) - 1).tolist()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = numpy.array(label).reshape(-1, 2)
            label = label[self.flip_index, :]
            label[:, 0] = 1 - label[:, 0]
            label = label.flatten()
            return image, label
        else:
            return image, label


class RandomCutOut:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = numpy.array(image).astype(numpy.uint8)
            image = image[:, :, ::-1]
            h, w, _ = image.shape
            cut_h = int(h * 0.4 * random.random())
            cut_w = int(w * 0.4 * random.random())
            x = int((w - cut_w - 10) * random.random())
            y = int((h - cut_h - 10) * random.random())
            image[y:y + cut_h, x:x + cut_w, 0] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 1] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 2] = int(random.random() * 255)
            image = Image.fromarray(image[:, :, ::-1].astype('uint8'), 'RGB')
            return image, label
        else:
            return image, label


class RandomGaussianBlur:
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            radius = random.random() * 5
            gaussian_blur = ImageFilter.GaussianBlur(radius)
            image = image.filter(gaussian_blur)
        return image, label


class RandomHSV:
    def __init__(self, h=0.015, s=0.700, v=0.400, p=0.500):
        self.h = h
        self.s = s
        self.v = v
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = numpy.array(image)
            r = numpy.random.uniform(-1, 1, 3)
            r = r * [self.h, self.s, self.v] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

            x = numpy.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype('uint8')
            lut_sat = numpy.clip(x * r[1], 0, 255).astype('uint8')
            lut_val = numpy.clip(x * r[2], 0, 255).astype('uint8')

            image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)
            image = Image.fromarray(image)
        return image, label


class RandomRGB2IR:
    """
    RGB to IR conversion
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            return image, label
        image = numpy.array(image)
        image = image.astype('int32')
        delta = numpy.random.randint(10, 90)

        ir = image[:, :, 2]
        ir = numpy.clip(ir + delta, 0, 255)
        return Image.fromarray(numpy.stack((ir, ir, ir), axis=2).astype('uint8')), label


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class ComputeLoss:
    def __init__(self, params):
        super().__init__()
        self.cls = params['cls']
        self.reg = params['reg']
        self.num_neighbor = params['num_nb']
        self.criterion_reg = torch.nn.L1Loss()
        self.criterion_cls = torch.nn.MSELoss()

    def __call__(self, outputs, targets):
        device = outputs[0].device
        b, c, h, w = outputs[0].size()

        score = outputs[0]
        offset_x = outputs[1].view(b * c, -1)
        offset_y = outputs[2].view(b * c, -1)
        neighbor_x = outputs[3].view(b * self.num_neighbor * c, -1)
        neighbor_y = outputs[4].view(b * self.num_neighbor * c, -1)

        target_score = targets[0].to(device).view(b * c, -1)
        target_offset_x = targets[1].to(device).view(b * c, -1)
        target_offset_y = targets[2].to(device).view(b * c, -1)
        target_neighbor_x = targets[3].to(device).view(b * self.num_neighbor * c, -1)
        target_neighbor_y = targets[4].to(device).view(b * self.num_neighbor * c, -1)

        target_max_index = torch.argmax(target_score, 1).view(-1, 1)
        target_max_index_neighbor = target_max_index.repeat(1, self.num_neighbor).view(-1, 1)

        offset_x_select = torch.gather(offset_x, 1, target_max_index)
        offset_y_select = torch.gather(offset_y, 1, target_max_index)
        neighbor_x_select = torch.gather(neighbor_x, 1, target_max_index_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, target_max_index_neighbor)

        target_offset_x_select = torch.gather(target_offset_x, 1, target_max_index)
        target_offset_y_select = torch.gather(target_offset_y, 1, target_max_index)
        target_neighbor_x_select = torch.gather(target_neighbor_x, 1, target_max_index_neighbor)
        target_neighbor_y_select = torch.gather(target_neighbor_y, 1, target_max_index_neighbor)

        loss_cls = self.criterion_cls(score, target_score.view(b, c, h, w))
        loss_offset_x = self.criterion_reg(offset_x_select, target_offset_x_select)
        loss_offset_y = self.criterion_reg(offset_y_select, target_offset_y_select)
        loss_neighbor_x = self.criterion_reg(neighbor_x_select, target_neighbor_x_select)
        loss_neighbor_y = self.criterion_reg(neighbor_y_select, target_neighbor_y_select)

        loss_cls = self.cls * loss_cls
        loss_reg = self.reg * (loss_offset_x + loss_offset_y + loss_neighbor_x + loss_neighbor_y)
        return loss_cls + loss_reg


def distance2box(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return numpy.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    outputs = []
    for i in range(0, distance.shape[1], 2):
        p_x = points[:, i % 2] + distance[:, i]
        p_y = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            p_x = p_x.clamp(min=0, max=max_shape[1])
            p_y = p_y.clamp(min=0, max=max_shape[0])
        outputs.append(p_x)
        outputs.append(p_y)
    return numpy.stack(outputs, axis=-1)


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x,
                                     1.0 / 128,
                                     input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = outputs[idx][0]
                boxes = outputs[idx + fmc][0]
                boxes = boxes * stride
            else:
                scores = outputs[idx]
                boxes = outputs[idx + fmc]
                boxes = boxes * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = numpy.stack(numpy.mgrid[:height, :width][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(numpy.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = numpy.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = numpy.where(scores >= score_thresh)[0]
            bboxes = distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list

    def detect(self, image, input_size=None, score_threshold=0.5, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if image_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = numpy.zeros((input_size[1], input_size[0], 3), dtype=numpy.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list = self.forward(det_img, score_threshold)

        scores = numpy.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = numpy.vstack(bboxes_list) / det_scale
        pre_det = numpy.hstack((bboxes, scores)).astype(numpy.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = numpy.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = numpy.sum(numpy.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            index = numpy.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
        return det

    def nms(self, outputs):
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = numpy.maximum(x1[i], x1[order[1:]])
            yy1 = numpy.maximum(y1[i], y1[order[1:]])
            xx2 = numpy.minimum(x2[i], x2[order[1:]])
            yy2 = numpy.minimum(y2[i], y2[order[1:]])

            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = numpy.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep
