# 实际地理图
import pandas as pd
import numpy as np
import os
import folium
from folium import plugins
#import webbrowser
import geopandas as gp
# 数据导入：
full = pd.read_excel(
    "D:\program\pythonPractice\mathModel\pythonDraw\dbscan_cluster.xlsx")
full = full.dropna()
# 创建地图对象：
schools_map = folium.Map(
    location=[full['lat'].mean(), full['lon'].mean()], zoom_start=10)
marker_cluster = plugins.MarkerCluster().add_to(schools_map)
# 标注数据点：
for name, row in full.iterrows():
    folium.Marker([row["lat"], row["lon"]]).add_to(marker_cluster)
# 逐行读取经纬度，数值，并且打点
#folium.RegularPolygonMarker([row["lat"], row["lon"]], popup="{0}:{1}".format(row["cities"], row["GDP"]),number_of_sides=10,radius=5).add_to(marker_cluster)
schools_map.save(
    'D:\program\pythonPractice\mathModel\pythonDraw\schools_map.html')  # 保存到本地

# webbrowser.open('schools_map.html') #在
