# import seisbench
import seisbench.models as sbm
from obspy import read
import matplotlib.pyplot as plt
import pandas as pd
import os

model = sbm.EQTransformer.from_pretrained("stead")
print(model.weights_docstring)

stream = read("../Data/0611Hualien.mseed")
annotations = model.annotate(stream)

fig = plt.figure(figsize=(15, 10))
axs = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0})

offset = annotations[0].stats.starttime - stream[0].stats.starttime
print(annotations)

# 建立儲存圖片的資料夾
output_dir = "seismic_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 每 3 條 trace 一組進行處理
for i in range(0, len(stream), 3):
    # 取得當前組別的 trace (i, i+1, i+2)
    current_traces = stream[i:i+3]
    current_annotations = annotations[i:i+3]
    
    # 取得測站名稱作為檔名
    station_name = current_traces[0].stats.station
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0})
    
    # 計算時間偏移量 (以該組第一條 trace 為準)
    offset = current_annotations[0].stats.starttime - current_traces[0].stats.starttime
    
    # 繪製原始波形與預測曲線
    for j in range(3):
        tr = current_traces[j]
        ann = current_annotations[j]
        
        # 上圖：原始波形
        axs[0].plot(tr.times(), tr.data, label=tr.stats.channel)
        
        # 下圖：機率曲線 (排除噪聲通道)
        if ann.stats.channel[-1] != "N":
            axs[1].plot(
                ann.times() + offset,
                ann.data,
                label=ann.stats.channel,
            )
            
    # 圖表裝飾
    axs[0].set_title(f"Station: {station_name}")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_ylabel("Probability")
    axs[1].set_xlabel("Time (s)")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    
    # 儲存圖片並關閉物件以節省記憶體
    save_path = os.path.join(output_dir, f"{station_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"已儲存測站 {station_name} 的圖表至 {save_path}")

print("--- 所有測站處理完畢 ---")


output = model.classify(stream)
print(output.picks)
print(output.detections)

# 處理 Picks (P 到時, S 到時)
# SeisBench 的 Pick 物件可以直接轉換為字典清單
picks_data = []
for p in output.picks:
    picks_data.append({
        "station": p.trace_id,
        "phase": p.phase,
        "peak_time": p.peak_time,
        "peak_value": p.peak_value,  # 模型預測的信心度
        "start_time": p.start_time,
        "end_time": p.end_time
    })

df_picks = pd.DataFrame(picks_data)

# 處理 Detections (事件偵測視窗)
detections_data = []
for d in output.detections:
    detections_data.append({
        "station": d.trace_id,
        "start_time": d.start_time,
        "end_time": d.end_time,
        "peak_value": d.peak_value
    })

df_detections = pd.DataFrame(detections_data)

# 儲存成檔案
# 建議分開存，或者存入同一個 Excel 的不同分頁
output_csv_picks = os.path.join(output_dir, "picks_results.csv")
output_csv_detections = os.path.join(output_dir, "detections_results.csv")

df_picks.to_csv(output_csv_picks, index=False, encoding='utf-8-sig')
df_detections.to_csv(output_csv_detections, index=False, encoding='utf-8-sig')

print(f"--- 數據整合完畢 ---")
print(f"Picks 已儲存至: {output_csv_picks} (共 {len(df_picks)} 筆)")
print(f"Detections 已儲存至: {output_csv_detections} (共 {len(df_detections)} 筆)")



