# MSNLAN
After the paper is accepted, we will provide code for the paper: Multi-Scale Non-Local Attention Network for Image Super-Resolution.
# Abstract
Natural images tend to recur similar patterns within the same scale and across different scales. Some recent progress on Single Image Super-Resolution (SISR) have elaborated on applying non-local attention mechanism to explore intrinsic feature correlation on multi-scale similarity to improve SR performance. However, recent advanced non-local attention-based methods typically utilize fixed-scale image patches to mine the correlations between local and global features, which is unreliable to judge the similarity between the non-local image patches across various scales. In this paper, we develop a novel multi-scale non-local attention network (MSNLAN) for SISR. The MSNLAN consists of a serial of Residual Groups (RGs) and an embedded Multi-Scale Non-local Attention Block (MSNLAB) located at the middle of those RGs. Each RG is comprised of several Residual Multi-scale Attention Blocks (RMAB) which learns multi-scale features extracted from different receptive fields to fully capture diverse and complementary feature representations. While MSNLAB is applied to establish more long-range dependencies by exploring multi-scale patches within the same feature map to seek more faithful and relevant non-local features. Moreover, we utilize Dense Skip Connections (DSC) between each RG and the MSNLAB to enhance the interaction of different context features, which is propitious for generating more impressive image details. Extensive experiments demonstrate that the proposed MSNLAN outperforms other state-of-the-art competitors in terms of both quantitative and qualitative quality assessment results.
# The overall framework
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/74ccbec4-9c0a-4a35-9044-8f6fe297a68c)
# Quantitative results
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/9a1f232f-fdc3-4340-a5e5-5b3e377e9e2a)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/2bebd90b-97be-4158-b394-aae6445c8103)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/93558b50-3794-4782-bfae-b7c2cf4f2339)
# Qualitive results
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/f68a47c6-3849-43f3-9eee-70f18285a89f)
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/1c25fec7-9aa2-42cf-9a1a-32f2f5984992)
# Model size comparison
![image](https://github.com/kbzhang0505/MSNLAN/assets/97494153/6c8605bb-9cc7-4feb-9bd5-bb037a1f58e0)

# Datasets Structure
For training, you need to build the new directory! In option.py, '--dir_data' based on the HR and LR images path. '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

*|——DIV2K

  **|——bin
  
  **|——DIV2K_train_HR
    
  **|——DIV2K_train_LR_bicubic
    
     ***|——x2
     ***|——x3
     ***|——x4
For training, you need to build the new directory! e.g.:

*|——Set5

  **|——GTmod12

  **|——LRbicx2

    ***|——x2
  
  **|——LRbicx3

    ***|——x3
 
  **|——LRbicx4

    ***|——x4
# Weights
The weights of student models are available at https://drive.google.com/drive/folders/19Yu9dSqN2yEsVeSifyzvjEDgZ-iOaSSY?usp=sharing

The code is based on https://github.com/SHI-Labs/Pyramid-Attention-Networks and https://github.com/sanghyun-son/EDSR-PyTorch.
      
# Thanks
    @article{mei2020pyramid,
      title={Pyramid Attention Networks for Image Restoration},
      author={Mei, Yiqun and Fan, Yuchen and Zhang, Yulun and Yu, Jiahui and Zhou, Yuqian and Liu, Ding and Fu, Yun and Huang, Thomas S and Shi, Honghui},
      journal={arXiv preprint arXiv:2004.13824},
      year={2020}
    }
    @InProceedings{Lim_2017_CVPR_Workshops,
      author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
      title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month = {July},
      year = {2017}
    }
# Citation
If you find the code helpful in your research or work, please cite the following paper:

    @article{WU2023109362,
       title = {Multi-scale non-local attention network for image super-resolution},
       journal = {Signal Processing},
       pages = {109362},
       year = {2023},
       issn = {0165-1684},
       doi = {https://doi.org/10.1016/j.sigpro.2023.109362},
       url = {https://www.sciencedirect.com/science/article/pii/S016516842300436X},
       author = {Xue Wu and Kaibing Zhang and Yanting Hu and Xin He and Xinbo Gao}}
    
With any questions, feel welcome to contact us
