# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

# import threading
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import *
# import cv2

# from __future__ import division, print_function, absolute_import
import warnings
import sys
import threading
import os
import time
warnings.filterwarnings('ignore')
# import MainWindow
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from ops.transforms import *
from ops.models import TSN
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import copy
import cv2
# from timeit import time
from PIL import Image, ImageDraw, ImageFont, ImageQt

from video import Video
from creat import Creat
from folder.model import model
from folder.model import transform
from folder.model import my_dict
from PIL import Image
import os
import torch
import numpy as np
from numpy.random import randint
from det import Predictor
from sort import *
import os
from nanodet.util import cfg, load_config, Logger
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
a = 0
MAXNUMPLUS1 = 16
MAXNUM = 15
minrecnum = 1
frame_num = 16
zhongduan_flag = 1
show_frames = []
whole_frame = []
t = 0
dengous = []
# self.software_info_textBrowser = []



def _sample_indices(lenth):
    """

    :param record: VideoRecord
    :return: list
    """
    average_duration = (lenth) // 8
    if average_duration > 0:
        offsets = np.multiply(list(range(8)), average_duration) + randint(average_duration,size=8)
    elif lenth > 8:
        offsets = np.sort(randint(lenth, size=8))
    else:
        offsets = np.zeros((8,),dtype='int8')
    return offsets


def inference(data, gui):
    input = torch.FloatTensor()
    for item in data:
        lenth = len(item)
        indices = _sample_indices(lenth)
        temp = []
        gui()
        for index in indices:
            temp.append(item[index])
        this = transform(temp)
        this = this.unsqueeze(0)
        input =torch.cat((input, this), 0)
    input = input.cuda()
    with torch.no_grad():
        gui()
        output = model(input)
        gui()
        # print(output)
        _, label = torch.max(output.data, 1)
    return [my_dict[int(i)] for i in label]
class Ui_MainWindow(object):
    # global self.software_info_textBrowser
    # def __init__(self):
    #     # QMainWindow.__init__(self)
    #     # MainWindow.Ui_MainWindow.__init__(self)
    #     # self.setupUi(self)
    #     self.pushButton.clicked.connect(self.on_video)
    #     self.pushButton1.clicked.connect(self.open_video)
    #     self.pushButton.setEnabled(False)
    #
    #     self.open_flag = True
    #     self.video_stream = cv2.VideoCapture(1)
    #     self.painter = QPainter(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1340,890)
        self.zero_zhongduan = 1
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 1000, 31))
        self.label.setObjectName("label")

        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(1154, 20, 180, 31))
        self.label2.setObjectName("label")

        self.software_info_textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        # self.software_info_textBrowser.setMinimumSize(QtCore.QSize(366, 169))
        # self.software_info_textBrowser.setMaximumSize(QtCore.QSize(366, 169))
        self.software_info_textBrowser.setObjectName("self.software_info_textBrowser")
        self.software_info_textBrowser.setGeometry(QtCore.QRect(1120, 50, 200, 720))
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(500, 790, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setGeometry(QtCore.QRect(50, 790, 93, 28))
        self.pushButton1.setObjectName("pushButton")
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(975, 790, 93, 28))
        self.pushButton2.setObjectName("pushButton")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(20, 50, 1080, 720))
        self.widget.setObjectName("widget")
        self.img_label = QtWidgets.QLabel(self.widget)
        self.img_label.setGeometry(QtCore.QRect(0, 0, 1080, 720))
        self.img_label.setObjectName("img_label")
        # self.img_label = QtWidgets.QLabel()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 713, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.file_window = QtWidgets.QMainWindow()
        # self.exit_action = QtWidgets.QAction(MainWindow)
        # self.exit_action.setObjectName("exit_action")
        # self.exit_action.triggered.connect(MainWindow.close)
        self.pushButton2.clicked.connect(self.myclose)

        self.pushButton.clicked.connect(self.on_video)
        self.pushButton1.clicked.connect(self.open_video)
        self.pushButton.setEnabled(False)
        self.textflag = 0
        self.open_flag = True
        self.video_stream = 0#cv2.VideoCapture(1)
        self.read_flag = True
        # self.painter = QPainter(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "危险行为识别系统"))
        self.label.setText(_translate("MainWindow", "请载入视频"))
        self.label2.setText(_translate("MainWindow", "危险行为"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton1.setText(_translate("MainWindow", "打开"))
        self.pushButton2.setText(_translate("MainWindow", "退出"))
        self.img_label.setText(_translate("MainWindow", "                                                            视频播放区域"))

    def myclose(self):
        self.zero_zhongduan = 0
        self.open_flag = 0
        cv2.waitKey(10)
        sys.exit(0)

    def open_video(self):
        global zhongduan_flag, show_frames, t, a, frame_num, dengous, whole_frame, minrecnum, MAXNUMPLUS1, MAXNUM
        a = 0 
        font = ImageFont.truetype('STFANGSO.TTF', 40)
        self.read_flag = True
        gui = QtGui.QGuiApplication.processEvents
        show_frames = []
        self.img_label.setPixmap(QPixmap(""))
        self.software_info_textBrowser.clear()
        dingshitime = 0.14
        max_id = 0
        def change_user2():
            global whole_frame
            whole_frame = []
        
        def change_user():
            global frame_num, whole_frame
            if self.read_flag == True:# and len(whole_frame) <= 1:
                #ret, frame = self.video_stream.read()
                #frame=cv2.flip(frame,-1)
                #whole_frame.append(frame)
                print('zhongduan', len(whole_frame))
                #self.read_flag = ret
            #print('这是中断,切换账号')
            
            last_len = len(show_frames)
            while self.open_flag:
                gui()
                _zxcv = 1
                #cv2.waitKey(1)

            if self.zero_zhongduan:
                t = threading.Timer(dingshitime, change_user)
                t.start()
            if len(show_frames) > 1:
                print(len(show_frames))
                gui()
                frame_num -= 1

                self.openimage(show_frames[0], frame_num, gui)
                gui()
                del show_frames[0]

            if frame_num <= MAXNUM:
                frame_num = MAXNUMPLUS1
                self.textflag = 1

                # gui()
                # cv2.waitKey(0)
        #path = os.path.abspath(__file__)
        #res = QFileDialog.getOpenFileName(self.file_window, 'open video', path,
        #                                  '*.mp4 *.avi')
        # yolo.a = 0
        # if res == None:
        #     self.video_stream = cv2.VideoCapture(1)
        _translate = QtCore.QCoreApplication.translate
        #s = '载入 ' + res[0] + '!'
        # print('??', res[0],'..')
        # if res[0] == '':
        #     s = 'load ' + 'None!'
        #self.label.setText(_translate("MainWindow", s))
        self.pushButton.setText('开始')

        gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(0, 1280, 720)
        self.video_stream = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


        #self.video_stream = cv2.VideoCapture(res[0])
        self.pushButton.setEnabled(True)
        self.pushButton1.setEnabled(False)
        ###########processing
        dengous = []
        max_recog_num = MAXNUM
        KalmanBoxTracker.count = 0
        mot_tracker = Sort()
        writeVideo_flag = True
        # path = 'E:\剪视频\数据第一波\投掷裁过/train4010.mp4'
        # now_path = os.getcwd('')
        # #now_path.replace("\\",'/')
        # if not os.path.exists(now_path+'\\'+'outputs'):
        #     os.mkdir('outputs')

        seq = 1
        w = int(self.video_stream.get(3))
        h = int(self.video_stream.get(4))
        # ret, frame = video_capture.read()
        if writeVideo_flag:
            # Define the codec and create VideoWriter object

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # out = cv2.VideoWriter('output.avi', fourcc, 8, (w, h))
            # list_file = open('detection.txt', 'w')
            frame_index = -1

        fps = 0.0

        recog_index = 0
        det_flag = 0
        videos = []
        whole_frame = []
        ranks_creat = []
        show_frames = []
        self.showflag = 0
        # zhongduan_flag = 0
        if zhongduan_flag:
            # gui()
            t = threading.Timer(dingshitime, change_user)
            t.start()
            # gui()
            zhongduan_flag = 0
        delflag = True
        while True:
            
            while self.open_flag:# or len(whole_frame) == 0:
                gui()
                _ = 1
                gui()
            # gui()
            #print('whole_frame_______', len(self.whole_frame))
            ret, frame = self.video_stream.read()#self.read_flag, whole_frame[0]  # frame shape 640*480*3
            #print('jiance', len(whole_frame))
            #del whole_frame[0]
            gui()
            #whole_frame.append(frame)
            if recog_index == max_recog_num:
                recog_index = 0
                videos = []
            recog_index += 1

            if ret != True:
                recog_index = 0
                break
            #t1 = time.time()

            # image = Image.fromarray(frame)
            dets = []
            if det_flag % 3 == 0 or det_flag % 3 == 1 or det_flag % 3 == 2:
            	res = predictor.inference(frame)
            	#print('jiance+++++++++++++', time.time() - t1)
            	res0 = res[0]
            	dets = []
            	for box in res0:
                	if box[4] > 0.35:
                    		dets.append(box)
            	dets = np.array(dets)
            det_flag += 1
            ########################track
            
            trackers = mot_tracker.update(dets)

            saveboxs = []
            saveboxstwo = []
            saveids = []
            flag = 0
            
            for track in trackers:
                bbox = track.astype(int)
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                #
                # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                """
                my first code
                """
                saveboxs.append(bbox)
                saveboxstwo.append([bbox[0],bbox[1],bbox[2],bbox[3]])
                saveids.append(bbox[4])
                max_id = max(max_id, bbox[4])
                flag = 1

            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if flag == 1:
                videos.append(Video(frame, saveids, saveboxs, saveboxstwo))
            if flag == 0:
                videos.append(Video(frame, [], [], []))


            #fps = time.time() - t1
            #print("fps= %f" % (fps))

            # Press Q to stop!
            # gui()
            #print('index', recog_index, max_recog_num)
            if recog_index == max_recog_num:
                t2 = time.time()
                gui()
                ranks_creat = []
                id_list = range((max_id - 10), (max_id + 1))
                for i in id_list:
                    box_list = []
                    rank_frames = []
                    gui()
                    for rank_videos in videos:
                        gui()
                        if i in rank_videos.id:
                            box_list.append(rank_videos.box[rank_videos.id.index(i)])
                            rank_frames.append(rank_videos.frame)
                    if (len(rank_frames)) > minrecnum:
                        gui()
                        ranks_creat.append(Creat(rank_frames, i, box_list))
                        gui()
                gui()
                classifier = []
                recog_id = []
                # self.dengous = []
                rec_frames_batch = []
                weizhis_batch = []
                classifiers = []
                #print('shijian========diyibufen', time.time() - t2)
                for max_creat in ranks_creat:
                    gui()
                    max_creat.max_box(w, h)
                    gui()
                    # print(max_creat.h, max_creat.w)

                    rec_frames = max_creat.creat_video(w, h, seq)
                    gui()
                    # for img in rec_frames:
                    #     cv2.imshow('a', img)
                    #     cv2.waitKey(0)
                    # start = time.time()
                    #########
                    # classifier.append(recognize(max_creat.recog_frame, model, n2w, transform))
                    # classifiers = ['quanda', 'jiaoti']  # 'leibie'#[quanda,jiaoti] nidehanshu(rec_frames)
                    if len(rec_frames) >= minrecnum:
                        #print('len',len(rec_frames))
                        rec_frames_batch.append(rec_frames)
                #print('shijian========dier', time.time() - t2)
                #for i in range(len(rec_frames_batch)):#nidehanshu(rec_frames_batch)#['quanda', 'jiaoti']
                #    classifiers.append('quanda')
                gui()
                if len(rec_frames_batch) != 0:                 
                    gui()
                    classifiers = inference(rec_frames_batch, gui)
                    gui()
                
                # print(classifiers)
                for i, classifier in enumerate(classifiers):
                    gui()
                    if ([ranks_creat[i].id, classifier] not in dengous) and (classifier != '正常'):
                        dengous.append([ranks_creat[i].id, classifier])
                        self.showflag += 1
                    gui()
                    for video in videos:
                        gui()
                        if ranks_creat[i].id in video.id:
                            video.id[video.id.index(ranks_creat[i].id)] = [video.id[video.id.index(ranks_creat[i].id)],
                                                                          classifier]
################################################################
                #print('shijian========disan', time.time() - t2)
                # if show_frames != []:
                #     t = threading.Timer(0.5, change_user)
                #     t.start()
                # cv2.waitKey(100)
                fillColor = (255, 255, 0)
                for video in videos:
                    gui()
                    #img_PIL = Image.fromarray(cv2.cvtColor(video.frame, cv2.COLOR_BGR2RGB))
                    img_PIL = video.frame
                    
                    
                    # fontsize = 40
                    # font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), fontsize)
                    for show_i in range(len(video.id)):
                        # cv2.rectangle(img_PIL, (int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][1])), (int(video.boxtwo[show_i][2]), int(video.boxtwo[show_i][3])),
                        #               (255, 255, 255), 2)
                        gui()
                        #print('byte============', len(video.id[show_i]))
                        try:
                            _ = (len(video.id[show_i]))
                        except:
                            video.id[show_i] = [video.id[show_i], '正常']
                        # if type(video.id[show_i]) != int:
                        #video.id[show_i][1] = video.id[show_i][1].encode('utf-8')

                        #video.id[show_i][1] = video.id[show_i][1].encode('utf-8').decode('utf8')

                        # videoshow = str(video.id[show_i]).encode('utf8')
                        gui()
                        position = (int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][1]))
                        gui()
                        draw = ImageDraw.Draw(img_PIL)
                        gui()
                        draw.text(position, str(video.id[show_i]), font=font, fill=fillColor)
                        gui()
                        # cv2.putText(img_PIL, str(video.id[show_i]), (int(video.boxtwo[show_i][0]),
                        # int(video.boxtwo[show_i][1])), 0, 5e-3 * 200,
                        #             (0, 255, 0), 2)
                        draw.line([(int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][1])),
                                   (int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][3]))], fill='yellow')
                        gui()
                        draw.line([(int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][1])),
                                   (int(video.boxtwo[show_i][2]), int(video.boxtwo[show_i][1]))], fill='yellow')
                        gui()
                        draw.line([(int(video.boxtwo[show_i][2]), int(video.boxtwo[show_i][1])),
                                   (int(video.boxtwo[show_i][2]), int(video.boxtwo[show_i][3]))], fill='yellow')
                        gui()
                        draw.line([(int(video.boxtwo[show_i][0]), int(video.boxtwo[show_i][3])),
                                   (int(video.boxtwo[show_i][2]), int(video.boxtwo[show_i][3]))], fill='yellow')
                        gui()
                    #tchange = time.time()
                    #video.frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                    #print('shijian+++++++++++++++quanchange', (time.time() - tchange) * 30)
                    gui()
                    show_frames.append(video.frame)
                    gui()
                # print('attend')
                videos = []
                #print('shijian========quanbu', time.time() - t2)

            if self.showflag and self.textflag:
                #print(dengous)
                dengou = dengous[a:]
                for action in dengou:
                    gui()
                    self.software_info_textBrowser.append('标号:' + str(action[0]) + '  行为:' + str(action[1]))
                    #print('标号:' + str(action[0]) + '  行为:' + str(action[1]))
                    a += 1
                    self.showflag -= 1
                self.textflag = 0
                if frame_num <= 1:
                    frame_num = MAXNUMPLUS1

            #t2 = threading.Timer(0.01, change_user2)
            #t2.start()
        ###########
        # self.video_stream = cv2.VideoCapture('output.avi')
        while len(show_frames) != 0:
            gui()
            cv2.waitKey(1)
            gui()
        self.label.setText(_translate("MainWindow", '识别结束'))
        self.pushButton.setText('完成')
        self.pushButton.setEnabled(False)
        self.pushButton1.setEnabled(True)
        self.open_flag = True
        self.dengous = []
    def on_video(self):
        if not self.open_flag:
            self.pushButton.setText('开始')
        else:
            self.pushButton.setText('暂停')
        self.open_flag = bool(1 - self.open_flag)  #

    def openimage(self, frame, num, gui):

        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        # jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        # frame = cv2.imread('E:\\xunleixiazai\imgnew\\0_{}.jpg'.format(a))
        # a += 1
        # base64_code = frame_to_base64(frame)
        #show = cv2.resize(frame, (1080, 720))
        show = frame.resize((1080, 720))
        # print(show)
        #show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        #show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        
        show_image = ImageQt.ImageQt(show)   
        # self.label.setPixmap(QtGui.QPixmap.fromImage(show_image))
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(show_image))
        gui()
        # if self.showflag and (num - 25) == 0:
        #     dengous = self.dengous[a:]
        #     for action in dengous:
        #         self.software_info_textBrowser.append('id:' + str(action[0]) + 'action:' + str(action[1]))
        #         # print('id:' + str(action[0]) + 'action:' + str(action[1]))
        #         a += 1
        #     self.showflag = 0
        # cv2.waitKey(100)
if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--demo', default='video', help='demo type, eg. image, video and webcam')
        parser.add_argument('--config', default='./config/nanodet-m.yml', help='model config file path')
        parser.add_argument('--model', help='model file path', default='nanodet_m_oldversion.pth')
        # parser.add_argument('--path', default='E:\\xunleixiazaitwo\\xiangmu\\fengzhuang\\touzhicaiguo\\train4001.mp4',
        #                     help='path to images or video')
        parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
        args = parser.parse_args()
        return args
    args = parse_args()
    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cuda:0')

    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
