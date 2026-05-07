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
MSEED_PATH = "../Data/0224_17_18.mseed"
OUTPUT_DIR = "seismic_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Gamma 參數設定 ---
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

# --- 3. 獲取時間範圍 ---
print("正在掃描檔案資訊...")
st_header = read(MSEED_PATH, headonly=True)
global_start = min(tr.stats.starttime for tr in st_header)
global_end = max(tr.stats.endtime for tr in st_header)
print(f"資料範圍: {global_start} ~ {global_end}")

# --- 4. 滑動視窗處理流程 ---
PICK_FILE = "./seismic_plots/picks_results.csv"
if os.path.exists(PICK_FILE):
    print(f"正在讀取拾取結果: {PICK_FILE}")

    # 讀取 CSV
    raw_pick_df = pd.read_csv(PICK_FILE)
    
    # 轉換格式以符合 GaMMA 需求
    # GaMMA 必備欄位: id, type, time, prob
    pick_df = pd.DataFrame({
        "id": raw_pick_df["station"],
        "type": raw_pick_df["phase"].str.lower(),   # 轉成小寫 p, s
        "timestamp": pd.to_datetime(raw_pick_df["peak_time"]), # 轉成 datetime 物件
        "prob": raw_pick_df["peak_value"]
    })
    
    # 去除重複項 (避免同一個點位被存入多次)
    pick_df = pick_df.drop_duplicates(subset=["id", "timestamp", "type"])
    
    print(f"✅ 成功讀取 {len(pick_df)} 個拾取點")
else:
    print(f"❌ 找不到檔案 {PICK_FILE}，請確認路徑。")
    pick_df = pd.DataFrame()

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

# --- 5. 輸出分析報告與詳細編目 ---

# A. 座標回推 (公里 -> 經緯度)
# Transformer 預期 (x, y) 單位通常是公尺，故需 * 1e3
if not catalog.empty:
    catalog["latitude"], catalog["longitude"] = transformer.transform(
        catalog["x(km)"] * 1e3, catalog["y(km)"] * 1e3, direction="INVERSE"
    )

# B. 整合資訊與判定類別
# 我們利用 assignments 來計算每個 event_idx 實際關聯到的站數
event_station_counts = assignments.groupby("event_idx")["pick_idx"].count()
catalog["total_picks"] = catalog["event_index"].map(event_station_counts)

def get_note(n):
    if n >= 5: return f"Earthquake (Total {n} picks)"
    elif 3 <= n < 5: return f"Suspected (Total {n} picks)"
    else: return f"Noise/Isolated (Total {n} picks)"

catalog["note"] = catalog["total_picks"].apply(get_note)

# C. 篩選與格式化輸出
# 為了符合你的格式，我們從 pick_df 中找回每個事件「第一個觸發的站點」作為代表
event_details = []
for _, row in catalog.iterrows():
    # 找到屬於該事件的所有 picks
    event_picks_indices = assignments[assignments["event_idx"] == row["event_index"]]["pick_idx"]
    event_picks = pick_df.iloc[event_picks_indices]
    first_pick = event_picks.sort_values("timestamp").iloc[0] # 取最早觸發的站

    event_details.append({
        "station": first_pick["id"],
        "start_time_str": row["time"],
        "lat": round(row["latitude"], 4),
        "lon": round(row["longitude"], 4),
        "depth_km": round(row["z(km)"], 2),
        "gamma_score": round(row["gamma_score"], 2),
        "sigma_time": round(row["sigma_time"], 3),
        "event_id": int(row["event_index"]),
        "total_picks": int(row["total_picks"]),
        "note": row["note"]
    })

detail_df = pd.DataFrame(event_details)

# D. 儲存檔案
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "association_report_G.txt")

# 1. 篩選資料
confirmed = detail_df[detail_df["total_picks"] >= 5]
suspected = detail_df[(detail_df["total_picks"] >= 3) & (detail_df["total_picks"] < 5)]

# 2. 定義要顯示的欄位
cols = ["station", "start_time_str", "lat", "lon", "depth_km", "gamma_score", "event_id", "note"]

# 3. 組合報告內容
report_lines = [
    "\n分析完成！",
    "-" * 40,
    f"原始觸發總數：{len(pick_df)} 筆",
    f"【確定】地震事件數 (>=5 站)：{len(confirmed)} 個",
    f"【疑似】地震事件數 (3-4 站)：{len(suspected)} 個",
    
    "\n--- 地震事件列表 (>=5 站) ---",
    confirmed[cols].to_string(index=False) if not confirmed.empty else "（無符合事件）",
    
    "\n--- 疑似地震列表 (3-4 站) ---",
    suspected[cols].to_string(index=False) if not suspected.empty else "（無符合事件）"
]

# 4. 合併並一次性輸出
full_report_text = "\n".join(report_lines)

# (選配) 如果想把這份純文字報告也存檔：
# with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
    # f.write(full_report_text)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(full_report_text)

print(f"\n[系統通知] 統計報告已同步儲存至：{OUTPUT_REPORT}")


import os
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import UTCDateTime

# --- 核心優化 1：僅讀取標頭以獲取基本資訊，不載入數據 ---
print("正在索引 mseed 檔案...")
st_header = obspy.read(MSEED_PATH, headonly=True)

# 1. 確保輸出目錄存在
EVENT_PLOT_DIR = os.path.join(OUTPUT_DIR, "event_plots")
os.makedirs(EVENT_PLOT_DIR, exist_ok=True)

print(f"開始繪製所有地震事件圖，總計：{len(catalog)} 個事件")

for _, event in catalog.iterrows():
    event_idx = int(event["event_index"])
    event_indices = assignments[assignments["event_idx"] == event_idx]["pick_idx"]
    event_picks = pick_df.iloc[event_indices].to_dict(orient="records")
    
    if not event_picks: continue

    origin_time = UTCDateTime(event["time"])
    # 計算繪圖時間範圍
    pick_times = [UTCDateTime(p["timestamp"]) for p in event_picks]
    last_pick = max(pick_times)
    origin_time = UTCDateTime(event["time"])
    
    # 定義切割範圍 (前 10 秒，後 15 秒)
    plot_start = origin_time - 10
    plot_end = last_pick + 15
    print(f"正在繪製事件 {event_idx}：發震時間 {origin_time}, 站數 {len(np.unique([p['id'] for p in event_picks]))}")

    # --- 1. 初始化收集容器 ---
    all_p_times, all_p_dists = [], []
    all_s_times, all_s_dists = [], []
    all_distances = []
    unique_stations_in_event = [p["id"] for p in event_picks]

    pad = 5
    # 讀取波形
    sub = obspy.read(MSEED_PATH, starttime=plot_start - pad, endtime=plot_end + pad)
    sub.detrend("demean").filter("bandpass", freqmin=2.0, freqmax=5.0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 2. 第一遍：繪製波形並收集所有 Pick 數據 ---
    for trace in sub:
        current_sta_id = f"{trace.stats.network}.{trace.stats.station}.10"
        if current_sta_id not in unique_stations_in_event: continue
        
        coords = station_dict.get(current_sta_id)
        if not coords: continue
        
        dist = np.sqrt((coords[0] - event["x(km)"])**2 + 
                       (coords[1] - event["y(km)"])**2 + 
                       (event["z(km)"] - coords[2] if len(coords)>2 else event["z(km)"])**2)
        all_distances.append(dist)

        # 收集該事件所有點位數據
        for p in event_picks:
            if p["id"] == current_sta_id:
                t_rel = UTCDateTime(p["timestamp"]) - origin_time
                if p["type"].lower() == "p":
                    all_p_times.append(t_rel); all_p_dists.append(dist)
                elif p["type"].lower() == "s":
                    all_s_times.append(t_rel); all_s_dists.append(dist)

        # 繪圖波形
        times = trace.times(reftime=origin_time)
        peak = np.max(np.abs(trace.data))
        if peak > 0:
            normed = (trace.data / peak) * 4
            ax.plot(times, normed + dist, lw=0.7, color="black", alpha=0.6)
            ax.text(-9, dist + 0.5, trace.stats.station, fontsize=7, color="blue", weight="bold")

        # 標註 Picks (Vertical Lines)
        for p in event_picks:
            if p["id"] == current_sta_id:
                x_pick = UTCDateTime(p["timestamp"]) - origin_time
                color = "blue" if p["type"].upper() == "P" else "red"
                ax.vlines(x_pick, dist - 3, dist + 3, color=color, lw=1.5, zorder=5)

    # --- 3. 第二遍：計算全局唯一的「平均/最大波速」 ---
    final_p_vel, final_s_vel = 0, 0
    if len(all_p_times) >= 2:
        # 線性擬合的斜率即為該事件的全局視速度
        final_p_vel = np.polyfit(all_p_times, all_p_dists, 1)[0]
    if len(all_s_times) >= 2:
        final_s_vel = np.polyfit(all_s_times, all_s_dists, 1)[0]

    # --- 4. 繪製 Legend (放在迴圈外，確保只出現一個) ---
    ax.axvline(0, color="green", lw=1.5, ls="--", label="Origin Time")
    
    if final_p_vel > 0:
        ax.plot([], [], color="blue", label=f"Max Avg Vp: {final_p_vel:.2f} km/s")
    if final_s_vel > 0:
        ax.plot([], [], color="red", label=f"Max Avg Vs: {final_s_vel:.2f} km/s")

    ax.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)
    
    # 調整圖表細節
    if all_distances:
        ax.set_ylim(min(all_distances) - 10, max(all_distances) + 10)
        ax.set_xlim(-10, (last_pick - origin_time) + 15)

    ax.set_ylabel("Hypocentral Distance (km)")
    ax.set_xlabel("Time (s) relative to Origin Time")
    ax.set_title(f"Event {event_idx} | Depth: {event['z(km)']:.2f}km")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(os.path.join(EVENT_PLOT_DIR, f"event_{event_idx:03d}.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)
    del sub