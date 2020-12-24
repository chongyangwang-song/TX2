import cv2
import numpy as np
from PIL import Image
import torchvision
import torch
from mobilenet_v2_tsm import MobileNetV2
SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True
import time
from ops.transforms import *

input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
normalize = GroupNormalize(input_mean, input_std)
transform = torchvision.transforms.Compose([
    GroupScale([457, 256]),
    GroupCenterCrop((400, 224)),
    Stack(roll=False),
    ToTorchFormatTensor(div=True),
    normalize,
])

# catigories = [
#     "Doing other things",  # 0
#     "Drumming Fingers",  # 1
#     "No gesture",  # 2
#     "Pulling Hand In",  # 3
#     "Pulling Two Fingers In",  # 4
#     "Pushing Hand Away",  # 5
#     "Pushing Two Fingers Away",  # 6
#     "Rolling Hand Backward",  # 7
#     "Rolling Hand Forward",  # 8
#     "Shaking Hand",  # 9
#     "Sliding Two Fingers Down",  # 10
#     "Sliding Two Fingers Left",  # 11
#     "Sliding Two Fingers Right",  # 12
#     "Sliding Two Fingers Up",  # 13
#     "Stop Sign",  # 14
#     "Swiping Down",  # 15
#     "Swiping Left",  # 16
#     "Swiping Right",  # 17
#     "Swiping Up",  # 18
#     "Thumb Down",  # 19
#     "Thumb Up",  # 20
#     "Turning Hand Clockwise",  # 21
#     "Turning Hand Counterclockwise",  # 22
#     "Zooming In With Full Hand",  # 23
#     "Zooming In With Two Fingers",  # 24
#     "Zooming Out With Full Hand",  # 25
#     "Zooming Out With Two Fingers"  # 26
# ]
my_dict = {0: '刺杀', 1: '射击', 2: '投掷', 3: '拳打',
           4: '掐脖子', 5: '脚踢', 6: '正常'}
catigories = ["stabing",
              "shooting",
              "throwing",
              "boxing",
              "squeeze_neck",
              "kicking",
              "normal"]
def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    # if idx_ in [7, 8, 21, 22, 3]:
    #     idx_ = history[-1]

    # use only single no action class
    # if idx_ == 0:
    #     idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

WINDOW_NAME = 'Video Gesture Recognition'

def main():
    # print("Open camera...")
    # gst_str = ('v4l2src device=/dev/video{} ! '
    #            'video/x-raw, width=(int){}, height=(int){} ! '
    #            'videoconvert ! appsink').format(0, 640, 480)
    cap = cv2.VideoCapture('/home/nvidia/Downloads/boxing_00002.avi')
    # cap = cv2.VideoCapture('/home/nvidia/Downloads/v_ApplyEyeMakeup_g01_c01.avi')
    print(cap)
    print("camera status:",cap.isOpened())
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    test_compile_path = '/home/nvidia/Documents/timeproject/ckpt.best.pth.tar'
    # net.load_state_dict()
    net = MobileNetV2(n_class=7)
    store = torch.load(test_compile_path)
    checkpoint = store['state_dict']
    new_checkpoint = {k.replace('module.base_model.', ''): v for k, v in checkpoint.items()}
    new_checkpoint = {k.replace('module.new_fc.', 'classifier.'): v for k, v in new_checkpoint.items()}
    new_checkpoint = {k.replace('net.', ''): v for k, v in new_checkpoint.items()}
    # new_checkpoint = {k.replace('base_model.',''):v for k,v in new_checkpoint.items()}
    net.load_state_dict(new_checkpoint)
    torch_module = net.cuda()
    torch_module.eval()
    buffer = [
        torch.zeros((1, 3, 100, 56), device='cuda'),
        torch.zeros((1, 4, 50, 28), device='cuda'),
        torch.zeros((1, 4, 50, 28), device='cuda'),
        torch.zeros((1, 8, 25, 14), device='cuda'),
        torch.zeros((1, 8, 25, 14), device='cuda'),
        torch.zeros((1, 8, 25, 14), device='cuda'),
        torch.zeros((1, 12, 25, 14), device='cuda'),
        torch.zeros((1, 12, 25, 14), device='cuda'),
        torch.zeros((1, 20, 13, 7), device='cuda'),
        torch.zeros((1, 20, 13, 7), device='cuda')
    ]


    idx = 0
    history = [2,3]
    history_logit = []
    i_frame = -1
    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            # img_tran = transform([img])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            # img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
            # inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            # outputs = executor(inputs)
            input_var = input_var.cuda()
            outputs = torch_module(input_var,*buffer)
            feat, buffer = outputs[0], outputs[1:]
            # assert isinstance(feat, tvm.nd.NDArray)

            if SOFTMAX_THRES > 0:
                feat_np = feat.numpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.detach().cpu().numpy(), axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat.detach().cpu().numpy(), axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat.detach().cpu().numpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            t2 = time.time()
            print(f"{index} {catigories[idx]}")

            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()

main()