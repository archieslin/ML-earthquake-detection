import seisbench
seisbench.use_backup_repository()

import seisbench.models as sbm
from obspy import read

# 測試載入模型
model = sbm.PhaseNet.from_pretrained("stead")
print(f"成功載入模型: {model.__class__.__name__}")

# 測試載入範例數據
try:
    import seisbench.data as sbd
    print("SeisBench 運作正常！")
except Exception as e:
    print(f"環境仍有問題: {e}")
