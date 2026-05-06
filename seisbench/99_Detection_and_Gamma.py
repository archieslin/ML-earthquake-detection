import os
import obspy
import json
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read
from pyproj import CRS, Transformer
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from gamma.utils import association
import seisbench.models as sbm

sns.set(font_scale=1.2)
sns.set_style("ticks")

# --- 1. 參數設定 ---
MSEED_PATH = "../Data/0502_14_15_HL.mseed"
OUTPUT_DIR = "seismic_plots"
CHUNK_LENGTH = 900  # 每次讀取 30 分鐘 (避免記憶體溢位)
# CHUNK_LENGTH = 60    # 每次讀取 1 分鐘 (單一事件)
OVERLAP = 60         # 時段間重疊 1 分鐘 (確保地震不被切斷)
STRIDE = 500         # 模型滑動步長 (500 samples = 5秒，重疊越高越準)
MODEL_TYPE = "stead" # 使用 EQTransformer

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1.1 Gamma 參數設定 ---
# Projections
wgs84 = CRS.from_epsg(4326)
local_crs = CRS.from_epsg(3826)  # SIRGAS-Chile 2016 / UTM zone 19S
transformer = Transformer.from_crs(wgs84, local_crs)

# Gamma
config = {}
config["dims"] = ["x(km)", "y(km)", "z(km)"]
config["use_dbscan"] = True
config["use_amplitude"] = False
config["x(km)"] = (100, 400)
config["y(km)"] = (2400, 2800)
config["z(km)"] = (0, 150)
config["vel"] = {
    "p": 5.8,
    "s": 2.5,
}  # We assume rather high velocities as we expect deeper events
config["method"] = "BGMM"
if config["method"] == "BGMM":
    config["oversample_factor"] = 4
if config["method"] == "GMM":
    config["oversample_factor"] = 1

# DBSCAN
config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # x
    (None, None),  # t
)
config["dbscan_eps"] = 25  # seconds
config["dbscan_min_samples"] = 3

# Filtering
config["min_picks_per_eq"] = 5
config["max_sigma11"] = 2.0
config["max_sigma22"] = 1.0
config["max_sigma12"] = 1.0

# --- 2. 初始化模型 ---
print(f"正在載入 {MODEL_TYPE} 模型...")
model = sbm.EQTransformer.from_pretrained(MODEL_TYPE)
# model.to("cuda") # 如果有 GPU 請取消註釋

# --- 3. 獲取時間範圍 ---
print("正在掃描檔案資訊...")
st_header = read(MSEED_PATH, headonly=True)
global_start = min(tr.stats.starttime for tr in st_header)
global_end = max(tr.stats.endtime for tr in st_header)
print(f"資料範圍: {global_start} ~ {global_end}")

all_picks_data = []
all_detections_data = []
pick_df = []

# --- 4. 滑動視窗處理流程 ---
current_start = global_start
while current_start < global_end:
    current_end = current_start + CHUNK_LENGTH
    print(f"\n>>> 處理時段: {current_start} ~ {current_end}")
    
    try:
        # 讀取區段資料ㄇ
        stream = read(MSEED_PATH, starttime=current_start, endtime=current_end + OVERLAP)
        if len(stream) == 0:
            current_start += CHUNK_LENGTH
            continue

        # --- 資料清理 (重要：解決 Fragments 與 Misalignment 警告) ---
        stream.detrend("demean")
        stream.merge(method=1, fill_value=0) # 接合斷點
        stream.trim(current_start, current_end + OVERLAP, pad=True, fill_value=0) # 對齊時間
        
        # 確保採樣率為 100Hz (EQT 標準)
        if stream[0].stats.sampling_rate != 100.0:
            stream.resample(100.0)

        # --- 模型推論 (自動執行 60s 視窗切片) ---
        # 透過 stride 控制 60s 視窗每次移動的距離
        annotations = model.annotate(stream, batch_size=24, stride=STRIDE)
        output = model.classify(stream, batch_size=24, stride=STRIDE)

        for p in output.picks:
            pick_df.append(
                {
                    "id": p.trace_id,
                    "timestamp": p.peak_time.datetime,
                    "prob": p.peak_value,
                    "type": p.phase.lower(),
                }
            )
        print(f"在 {current_start} ~ {current_end} 期間檢測到 {len(pick_df)} 個拾取點")
    except Exception as e:
        print(f"區段 {current_start} 處理失敗: {e}")

    current_start += CHUNK_LENGTH

# 迴圈結束後再轉為pandas DataFrame，避免每次迴圈都轉換造成效能問題
pick_df = pd.DataFrame(pick_df)

with open('stations.json', 'r', encoding='utf-8') as f:
    stations_data = json.load(f)

station_df_list = []
for station_name, info in stations_data.items():
    # 根據範例：coords[0]=Lat, coords[1]=Lon, coords[2]=Elv
    station_df_list.append({
        "id": f"{info['network']}.{station_name}.10",
        "longitude": info['coords'][1],  
        "latitude": info['coords'][0],   
        "elevation(m)": info['coords'][2],
    })

station_df = pd.DataFrame(station_df_list)

station_df["x(km)"] = station_df.apply(
    lambda x: transformer.transform(x["latitude"], x["longitude"])[0] / 1e3, axis=1
)
station_df["y(km)"] = station_df.apply(
    lambda x: transformer.transform(x["latitude"], x["longitude"])[1] / 1e3, axis=1
)
station_df["z(km)"] = -station_df["elevation(m)"] / 1e3

northing = {station: y for station, y in zip(station_df["id"], station_df["y(km)"])}
station_dict = {
    station: (x, y)
    for station, x, y in zip(station_df["id"], station_df["x(km)"], station_df["y(km)"])
    }

# Gamma 關聯分析
catalogs, assignments = association(pick_df, station_df, config, method=config["method"])
catalog = pd.DataFrame(catalogs)
assignments = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_aspect("equal")

# 先畫測站
ax.plot(station_df["x(km)"], station_df["y(km)"], "r^", mew=1, mec="k")

# 再畫震央
cb = ax.scatter(catalog["x(km)"], catalog["y(km)"], c=catalog["z(km)"], s=400, cmap="viridis")
print(len(catalog), "events detected by Gamma")
cbar = fig.colorbar(cb)
cbar.ax.set_ylim(cbar.ax.get_ylim()[::-1])
cbar.set_label("Depth[km]")
for x, y, t in zip(catalog["x(km)"], catalog["y(km)"], catalog["time"]):
        # 格式化時間：將 '2026-05-02T14:22:34.185' 轉換為 '14:22:34' 
        # 如果 time 是字串，取 [11:19]；如果是 datetime 物件，用 strftime
        t_label = t.strftime('%H:%M:%S') if hasattr(t, 'strftime') else str(t)[11:19]
        
        ax.text(
            x+10, y, 
            f"  {t_label}", # 前面加空格讓標籤與點位錯開
            fontsize=10, 
            ha="left",      # 水平對齊：靠左 (標籤在點的右邊)
            va="center",    # 垂直對齊：置中
            color="black", 
            weight="bold",
            zorder=4        # 確保標籤在最上層
        )
    

ax.set_xlabel("Easting [km]")
ax.set_ylabel("Northing [km]")
ax.set_xlim(-100, 400)
ax.set_ylim(2300, 3000)
plt.savefig("Gamma_Association_Result.png", bbox_inches="tight")
plt.close(fig)


