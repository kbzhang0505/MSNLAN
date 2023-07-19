# MSNLAN
After the paper is accepted, We will provide code for the Paper: Multi-Scale Non-Local Attention Network for Image Super-resolution.

Natural images tend to recur similar patterns within the same scale and across different scales. Some recent progress on Single Image Super-Resolution (SISR) have elaborated on applying non-local attention mechanism to explore intrinsic feature correlation on multi-scale similarity to improve SR performance. However, recent advanced non-local attention-based methods typically utilize fixed-scale image patches to mine the correlations between local and global features, which is unreliable to judge the similarity between the non-local image patches across various scales. In this paper, we develop a novel multi-scale non-local attention network (MSNLAN) for SISR. The MSNLAN consists of a serial of Residual Groups (RGs) and an embedded Multi-Scale Non-local Attention Block (MSNLAB) located at the middle of those RGs. Each RG is comprised of several Residual Multi-scale Attention Blocks (RMAB) which learns multi-scale features extracted from different receptive fields to fully capture diverse and complementary feature representations. While MSNLAB is applied to establish more long-range dependencies by exploring multi-scale patches within the same feature map to seek more faithful and relevant non-local features. Moreover, we utilize Dense Skip Connections (DSC) between each RG and the MSNLAB to enhance the interaction of different context features, which is propitious for generating more impressive image details. Extensive experiments demonstrate that the proposed MSNLAN outperforms other state-of-the-art competitors in terms of both quantitative and qualitative quality assessment results.
# Results
![image](https://github.com/against-wu/MSNLAN/assets/76865736/2f7c457e-25da-4be5-9437-b8a4998f92b4)
![image](https://github.com/against-wu/MSNLAN/assets/76865736/3b30469e-f61a-43ef-b5ad-11ff506d91e4)
![image](https://github.com/against-wu/MSNLAN/assets/76865736/41360de5-d9e1-46d9-a58b-90f50d9ca3f7)

![image](https://github.com/against-wu/MSNLAN/assets/76865736/f40466eb-cfd4-438b-a8c1-1bd7c13644fb)

