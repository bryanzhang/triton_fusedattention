参照triton官方tutorial实现fp16 fused attetnion推理, 已实现flashattention v2。
在A10上benchmark如下：
fused-attention-batch4-head32-d64:
     N_CTX  Triton [FP16]  Flash-v2 [FP16]  pytorch [FP16]
0   1024.0      63.736668        38.724957       54.526948
1   2048.0      67.337190        54.662378       59.492865
2   4096.0      70.019671        52.897926       63.965311
3   8192.0      73.170689        53.502509       66.557268
4  16384.0      74.099627        52.813033       64.442554

大致比cuda的flashattention v2快40%、比pytorch2实现快15%。
