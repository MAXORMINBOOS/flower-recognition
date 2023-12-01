import os
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
from sqlalchemy import create_engine, Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.mysql import LONGBLOB
Base = declarative_base()


class UserEvaluation(Base):
    __tablename__ = 'user_evaluations'
    image_filename = Column(String(255), primary_key=True)
    predicted_result = Column(String(255))
    user_evaluation = Column(Boolean)
    image_data = Column(LONGBLOB)


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('花卉识别')
        self.model = tf.keras.models.load_model("models/cnn_flower2.h5")
        self.to_predict_name = "images/init.png"
        self.class_names = ['Lilly', 'Lotus', 'Orchid', 'daisy', 'dandelion', 'green-glass', 'roses', 'sunflowers',
                            'tulips']
        self.resize(1000, 1000)
        self.initUI()
        self.engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/flower_evaluation',
                                    echo=True)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("测试样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)

        self.img_label.setPixmap(QPixmap('images/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 加载测试样本 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)

        btn_predict = QPushButton(" 识 别 花 卉 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        btn_evaluate = QPushButton(" 评价结果 ")
        btn_evaluate.setFont(font)
        btn_evaluate.clicked.connect(self.evaluate_result)
        right_layout.addWidget(btn_evaluate)

        label_result = QLabel(' 识 别 结 果 ')
        self.result = QLabel("待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        right_layout.addWidget(btn_evaluate)
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用花卉识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/logo.png'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页面')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            # 使用os.path.normpath确保路径格式正确
            self.to_predict_name = os.path.normpath(img_name)
            img_init = cv2.imread(self.to_predict_name)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap('images/target.png'))

    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        # gray_img = img.convert('L')
        # img_torch = self.transform(gray_img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        # print(outputs)
        result_index = np.argmax(outputs)
        # print(result_index)
        result = self.class_names[result_index]
        self.result.setText(result)

    def evaluate_result(self):
        try:
            # 检查是否已存在相同的记录
            existing_entry = self.session.query(UserEvaluation).filter_by(image_filename=self.to_predict_name).first()

            if existing_entry:
                # 弹出询问对话框，询问用户是否要更新评价
                update_result = QMessageBox.question(self, '更新评价', '您已经评价过这张图片，是否要更新评价？',
                                                     QMessageBox.Yes | QMessageBox.No)

                if update_result == QMessageBox.Yes:
                    # 在用户同意更新评价时再次询问用户对该图像的识别结果是否正确做出评价
                    self.show_evaluation_dialog(existing_entry.predicted_result)

                else:
                    QMessageBox.warning(self, '取消更新', '您取消了更新评价。')
            else:
                # 获取用户评价，这里使用QMessageBox作为示例
                result = QMessageBox.question(self, '评价', '识别结果是否正确？', QMessageBox.Yes | QMessageBox.No)

                if result == QMessageBox.Yes:
                    user_evaluation = True
                else:
                    user_evaluation = False

                    # 读取图像二进制数据
                with open(self.to_predict_name, 'rb') as f:
                    image_data = f.read()

                # 将评价结果写入数据库
                evaluation_entry = UserEvaluation(image_filename=self.to_predict_name,
                                                  predicted_result=self.result.text(),
                                                  user_evaluation=user_evaluation,
                                                  image_data=image_data)

                self.session.add(evaluation_entry)
                self.session.commit()
                QMessageBox.information(self, '成功', '谢谢你的评价！')

        except SQLAlchemyError as e:
            print(f"Error:{e}")
            self.session.rollback()
        finally:
            self.session.close()

    def show_evaluation_dialog(self, predicted_result):
        # 创建一个新的窗口来询问用户对该图像的识别结果是否正确做出评价
        evaluation_dialog = QDialog(self)
        evaluation_dialog.setWindowTitle('评价')
        evaluation_dialog.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        label_predicted_result = QLabel(f'预测结果：{predicted_result}')
        label_evaluation = QLabel('识别结果是否正确？')

        btn_correct = QPushButton('正确')
        btn_incorrect = QPushButton('不正确')

        btn_correct.clicked.connect(lambda: self.handle_evaluation_dialog(True, evaluation_dialog))
        btn_incorrect.clicked.connect(lambda: self.handle_evaluation_dialog(False, evaluation_dialog))

        layout.addWidget(label_predicted_result)
        layout.addWidget(label_evaluation)
        layout.addWidget(btn_correct)
        layout.addWidget(btn_incorrect)

        evaluation_dialog.setLayout(layout)

        # 显示新窗口
        evaluation_dialog.exec_()

    def handle_evaluation_dialog(self, is_correct, evaluation_dialog):
        # 处理用户对该图像的识别结果是否正确的评价
        user_evaluation = is_correct

        try:
            # 检查是否已存在相同的记录
            existing_entry = self.session.query(UserEvaluation).filter_by(image_filename=self.to_predict_name).first()

            if existing_entry:
                # 更新数据库中的评价记录
                existing_entry.user_evaluation = user_evaluation
                self.session.commit()
                QMessageBox.information(self, '成功', '评价已更新！')
            else:
                QMessageBox.warning(self, '错误', '未找到相应记录。')

        except SQLAlchemyError as e:
            print(f"Error:{e}")
            self.session.rollback()
        finally:
            self.session.close()

        # 关闭评价对话框
        evaluation_dialog.accept()


if __name__ == "__main__":
    sys.argv = [sys.argv[0].encode('uft-8')]
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
