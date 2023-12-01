import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


# 数据加载，按照8:2的比例加载花卉数据
def data_load(data_dir, img_height, img_width, batch_size):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = val_ds.class_names

    return val_ds, class_names


def evaluate_model(model, val_ds, transfer=True):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    model.summary()

    # 评估模型
    loss, accuracy, precision, recall = model.evaluate(val_ds)
    print('Test accuracy:', accuracy)
    print('Test precision:', precision)
    print('Test recall:', recall)

    # 预测验证集
    predictions = model.predict(val_ds)

    # 使用 map 函数获取标签
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)

    y_pred = np.argmax(predictions, axis=1)
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 使用 seaborn 绘制混淆矩阵图像
    if transfer == True:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Transfer Learning Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig("./models/TransferLearningCM2.png")
        plt.show()
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('CNN Learning Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig("./models/CNNLearningCM2.png")
        plt.show()


if __name__ == '__main__':
    val_ds, class_names = data_load("./data/flower_photos_split/val", 224, 224, 4)

    # 加载预训练好的模型
    transfer_model = tf.keras.models.load_model("models/mobilenet_flower2.h5")
    cnn_model = tf.keras.models.load_model("models/cnn_flower2.h5")

    # 评估模型并绘制混淆矩阵
    evaluate_model(transfer_model, val_ds, transfer=True)
    evaluate_model(cnn_model, val_ds, transfer=False)
