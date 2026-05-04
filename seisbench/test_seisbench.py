import seisbench
import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace, UTCDateTime

seisbench.use_backup_repository()

data = sbd.ETHZ(sampling_rate=100)
generator = sbg.GenericGenerator(data)

print(generator)

print("Number of examples:", len(generator))
sample = generator[200]
print("Example:", sample)

plt.plot(sample["X"].T);
plt.savefig("example.png")

generator.augmentation(sbg.RandomWindow(windowlen=3000))
generator.augmentation(sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1))

print(generator)

# 假設我們要標註 P 波與 S 波的位置
generator.label_generator = sbg.ProbabilisticLabeller(
    label_columns=["trace_p_arrival_sample", "trace_s_arrival_sample"],
    sigma=20  # 高斯標準差
)

sample = generator[200]
print("Labels shape:", sample["Y"].shape)