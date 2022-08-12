import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xml.dom.minidom as xmldom
import os

root = os.getcwd()

def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

def load_data(data_root="dataset/Annotations/"):
    h_w_ratios = []
    for i in find_all_file(data_root):
        xml_file = xmldom.parse(os.path.join(data_root, i))  # 读取xml文件
        elements = xml_file.documentElement  # 获取xml文件中的元素
        # 获得标签
        xmin = float(elements.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = float(elements.getElementsByTagName("ymin")[0].firstChild.data)
        xmax = float(elements.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = float(elements.getElementsByTagName("ymax")[0].firstChild.data)
        # 计算长宽比并保存
        h_w_ratios.append((ymax - ymin) / (xmax - xmin))
    return np.array(h_w_ratios), len(h_w_ratios)

def anchor_cluster(n_clusters=3):
    data_root = os.path.join(root, "dataset", "Annotations")
    data, _ = load_data(data_root)  # 读取长宽比数据
    data = data.reshape((-1, 1))

    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    n_clusters = 3
    aspect_ratio = anchor_cluster(n_clusters=n_clusters)



