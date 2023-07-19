# MSNLAN
After the paper is accepted, We will provide code for the Paper: Multi-Scale Non-Local Attention Network for Image Super-resolution.
# Abstract
Natural images tend to recur similar patterns within the same scale and across different scales. Some recent progress on Single Image Super-Resolution (SISR) have elaborated on applying non-local attention mechanism to explore intrinsic feature correlation on multi-scale similarity to improve SR performance. However, recent advanced non-local attention-based methods typically utilize fixed-scale image patches to mine the correlations between local and global features, which is unreliable to judge the similarity between the non-local image patches across various scales. In this paper, we develop a novel multi-scale non-local attention network (MSNLAN) for SISR. The MSNLAN consists of a serial of Residual Groups (RGs) and an embedded Multi-Scale Non-local Attention Block (MSNLAB) located at the middle of those RGs. Each RG is comprised of several Residual Multi-scale Attention Blocks (RMAB) which learns multi-scale features extracted from different receptive fields to fully capture diverse and complementary feature representations. While MSNLAB is applied to establish more long-range dependencies by exploring multi-scale patches within the same feature map to seek more faithful and relevant non-local features. Moreover, we utilize Dense Skip Connections (DSC) between each RG and the MSNLAB to enhance the interaction of different context features, which is propitious for generating more impressive image details. Extensive experiments demonstrate that the proposed MSNLAN outperforms other state-of-the-art competitors in terms of both quantitative and qualitative quality assessment results.
# Quantitative results
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/9a1f232f-fdc3-4340-a5e5-5b3e377e9e2a)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/2bebd90b-97be-4158-b394-aae6445c8103)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/93558b50-3794-4782-bfae-b7c2cf4f2339)
# Qualitive results
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/f68a47c6-3849-43f3-9eee-70f18285a89f)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/1c25fec7-9aa2-42cf-9a1a-32f2f5984992)
# Model size comparison
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/6c8605bb-9cc7-4feb-9bd5-bb037a1f58e0)

