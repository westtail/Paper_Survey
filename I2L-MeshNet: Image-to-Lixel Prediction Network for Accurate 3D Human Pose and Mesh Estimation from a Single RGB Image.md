## Title
I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image

I2L-MeshNet: 正確な3D人物像のための画像-画素間予測ネットワーク 1枚のRGB画像から人間の3Dポーズとメッシュを正確に推定する メッシュ推定

## Abstract.
Most of the previous image-based 3D human pose and mesh estimation methods estimate parameters of the human mesh model from an input image.   
However, directly regressing the parameters from the input image is a highly non-linear mapping because it breaks the spatial relationship between pixels in the input image.   
In addition, it cannot model the prediction uncertainty, which can make training harder.   
To resolve the above issues, we propose I2L-MeshNet, an image-to-lixel (line+pixel) prediction network.   
The proposed I2L-MeshNet predicts the per-lixel likelihood on 1D heatmaps for each mesh vertex coordinate instead of directly regressing the parameters.  
Our lixel-based 1D heatmap preserves the spatial relationship in the input image and models the prediction uncertainty.   
We demonstrate the benefit of the image-to-lixel prediction and show that the proposed I2L-MeshNet outperforms previous methods.   
The code is publicly available 1. 

これまでの画像ベースの3D人体姿勢・メッシュ推定法の多くは、入力画像から人体メッシュモデルのパラメータを推定するものでした。  
しかし、入力画像からパラメータを直接回帰することは、入力画像のピクセル間の空間的な関係を壊してしまうため、非常に非線形なマッピングとなります。  
また，予測の不確実性をモデル化することができないため，学習が困難になる可能性があります．  
このような問題を解決するために，我々は画像からリクセル（線＋画素）への予測ネットワークであるI2L-MeshNetを提案する．  
提案するI2L-MeshNetは、パラメータを直接回帰するのではなく、各メッシュ頂点座標の1次元ヒートマップ上で、リクセルごとの尤度を予測する。  
リクセルベースの1次元ヒートマップは、入力画像の空間的な関係を保持し、予測の不確実性をモデル化する。  
我々は，画像からリクセルへの予測の利点を示し，提案するI2L-MeshNetが以前の手法よりも優れていることを示す．  
コードは公開されています1。


## Introduction

3D human pose and mesh estimation aims to simultaneously recover 3D semantic human joint and 3D human mesh vertex locations.   
This is a very challenging task because of complicated human articulation and 2D-to-3D ambiguity.  
It can be used in many applications such as virtual/augmented reality and human action recognition.  

人間の3Dポーズとメッシュの推定は、人間の3Dセマンティックジョイントと3Dメッシュの頂点位置を同時に復元することを目的としています。  
これは、人間の複雑なアーティキュレーションや、2Dから3Dへの曖昧さのため、非常に困難なタスクです。   
この技術は、バーチャル／拡張現実感や人間の行動認識など、多くのアプリケーションに利用することができます。 

SMPL [28] and MANO [42] are the most widely used parametric human body and hand mesh models respectively which can represent various human poses and identities.   
They produce 3D human joint and mesh coordinates from pose and identity parameters.   
Recent deep convolutional neural network (CNN)-based studies [20, 23, 40] for the 3D human pose and mesh estimation are based on the model-based approach, which trains a network to estimate SMPL/MANO parameters from an input image.   
On the other hand, there have been few methods based on model-free approach [9, 24], which estimates mesh vertex coordinates directly.   
They obtain the 3D pose by multiplying a joint regression matrix, included in the human mesh model, to the estimated mesh.

SMPL[28]とMANO[42]は，それぞれ最も広く使われているパラメトリックな人体と手のメッシュモデルで，様々な人間のポーズやアイデンティティを表現することができます．   
これらのモデルは、ポーズとアイデンティティのパラメータから、人間の3次元関節とメッシュの座標を生成します。  
最近の深層畳み込みニューラルネットワーク（CNN）ベースの研究[20, 23, 40]は、モデルベースのアプローチに基づいており、入力画像からSMPL/MANOパラメータを推定するようにネットワークを学習させている。  
その一方で，メッシュの頂点座標を直接推定するモデルフリー・アプローチに基づく手法もいくつか存在する[9, 24]．   
これらの手法では，人間のメッシュモデルに含まれる関節回帰行列を，推定されたメッシュに乗じることで，3Dポーズを得ている．

Although the recent deep CNN-based methods perform impressive, when estimating the target (i.e., SMPL/MANO parameters or mesh vertex coordinates), all of the previous 3D human pose and mesh estimation works break the spatial relationship among pixels in the input image because of the fullyconnected layers at the output stage.   
In addition, their target representations cannot model the uncertainty of the prediction.   
The above limitations can make training harder, and as a result, reduce the test accuracy as addressed in [32,45].   
To address the limitations, recent state-of-the-art 3D human pose estimation methods [32, 33, 44], which localize 3D human joint coordinates without mesh vertex coordinates, utilize the heatmap as the target representation of their networks.   
Each value of one heatmap represents the likelihood of the existence of a human joint at the corresponding pixel positions of the input image and discretized depth value.   
Therefore, it preserves the spatial relationship between pixels in the input image and models the prediction uncertainty.

最近のディープCNNベースの手法は、ターゲット（SMPL/MANOパラメータやメッシュの頂点座標など）を推定する際には素晴らしい性能を発揮しますが、これまでの3D人間のポーズやメッシュの推定手法は、出力段階で完全に接続されたレイヤーがあるため、入力画像のピクセル間の空間的な関係を壊してしまいました。  
さらに、これらのターゲット表現は、予測の不確実性をモデル化することができません。  
上記の制限により，学習が困難になり，その結果，[32,45]で述べられているように，テストの精度が低下します．  
このような限界に対処するために、メッシュの頂点座標を使わずに人間の3次元関節座標をローカライズする最近の最先端の3次元人物姿勢推定法[32,33,44]では、そのネットワークのターゲット表現としてヒートマップを利用している。  
1つのヒートマップの各値は、入力画像の対応するピクセル位置と離散化された深度値における人間の関節の存在可能性を表している。  
そのため、入力画像のピクセル間の空間的な関係を保持し、予測の不確実性をモデル化することができます。

Inspired by the recent state-of-the-art heatmap-based 3D human pose estimation methods, we propose I2L-MeshNet, image-to-lixel prediction network that naturally extends heatmap-based 3D human pose to heatmap-based 3D human pose and mesh.    
Likewise voxel (volume+pixel) is defined as a quantized cell in three-dimensional space, we define lixel (line+pixel) as a quantized cell in one-dimensional space.   
Our I2L-MeshNet estimates per-lixel likelihood on 1D heatmaps for each mesh vertex coordinates, therefore it is based on the model-free approach.   
The previous state-of-the-art heatmap-based 3D human pose estimation methods predict 3D heatmap of each human joint.   
Unlike the number of human joints, which is around 20, the number of mesh vertex is much larger (e.g., 6980 for SMPL and 776 for MANO).   
As a result, predicting 3D heatmaps of all mesh vertices becomes computationally infeasible, which is be- yond the limit of modern GPU memory.   
In contrast, the proposed lixel-based 1D heatmap has an efficient memory complexity, which has a linear relationship with the heatmap resolution.   
Thus, it allows our system to predict heatmaps with sufficient resolution, which is essential for dense mesh vertex localization.

最近の最先端のヒートマップベースの3D人体姿勢推定法にヒントを得て、ヒートマップベースの3D人体姿勢をヒートマップベースの3D人体姿勢とメッシュに自然に拡張する画像-リクセル予測ネットワーク、I2L-MeshNetを提案する。   
3次元空間の量子化されたセルとしてvoxel(volume+pixel)が定義されているように、1次元空間の量子化されたセルとしてlixel(line+pixel)が定義されています。  
当社のI2L-MeshNetは、各メッシュの頂点座標に対する1次元ヒートマップを用いて、リクセルごとの尤度を推定するため、モデルフリーの手法を採用しています。  
これまでのヒートマップベースの3次元人物姿勢推定法は、人間の各関節の3次元ヒートマップを予測していました。  
しかし、人間の関節の数が20個程度であるのに対し、メッシュの頂点の数ははるかに多い（例えば、SMPLでは6980個、MANOでは776個）。  
その結果，すべてのメッシュ頂点の3Dヒートマップを予測することは，計算上不可能となり，最新のGPUメモリの限界を超えてしまいます．  
これに対して，提案されているリクセルベースの1次元ヒートマップは，メモリ使用量が効率的で，ヒートマップの解像度と線形関係にあります．  
そのため，本システムでは，高密度メッシュの頂点定位に必要な十分な解像度のヒートマップを予測することができます．

For more accurate 3D human pose and mesh estimation, we design the I2LMeshNet as a cascaded network architecture, which consists of PoseNet and MeshNet.   
The PoseNet predicts the lixel-based 1D heatmaps of each 3D human joint coordinate.   
Then, the MeshNet utilizes the output of the PoseNet as an additional input along with the image feature to predict the
lixel-based 1D heatmaps of each 3D human mesh vertex coordinate.   
As the locations of the human joints provide coarse but important information about the human mesh vertex locations, utilizing it for 3D mesh estimation is natural and can increase accuracy substantially.

人間の姿勢とメッシュをより正確に推定するために、I2LMeshNetはPoseNetとMeshNetからなるカスケードネットワークアーキテクチャとして設計されています。  
PoseNetは、人間の各関節座標のピクセルベースの1次元ヒートマップを予測します。  
そしてMeshNetは、PoseNetの出力を画像特徴量の追加入力として利用して、Lixelベースの1Dヒートマップを予測します。
リクセルベースの1次元ヒートマップを作成します。  
関節の位置は 人間の関節の位置は、人間のメッシュの頂点の位置について、粗いながらも重要な情報を提供します。頂点の位置に関する粗いながらも重要な情報を提供しているため、それを3Dメッシュ推定に利用することは自然なことであり、精度も大幅に向上します。精度が大幅に向上します。

Our I2L-MeshNet outperforms previous 3D human pose and mesh estimation methods on various 3D human pose and mesh benchmark datasets.   
Figure 1 shows 3D human body and hand mesh estimation results on publicly available datasets.  
Our contributions can be summarized as follows.
* We propose I2L-MeshNet, a novel image-to-lixel prediction network for 3D human pose and mesh estimation from a single RGB image. Our system predicts lixel-based 1D heatmap that preserves the spatial relationship in the input image and models the uncertainty of the prediction.
* Our efficient lixel-based 1D heatmap allows our system to predict heatmaps with sufficient resolution, which is essential for dense mesh vertex localization.
* We show that our I2L-MeshNet outperforms previous state-of-the-art methods on various 3D human pose and mesh datasets.

我々のI2L-MeshNetは、様々な3D人間のポーズとメッシュのベンチマークデータセットにおいて、これまでの3D人間のポーズとメッシュの推定手法よりも優れている。様々な3D人間のポーズとメッシュのベンチマークデータセットにおいて、これまでの3D人間のポーズとメッシュの推定手法を凌駕しています。  
図1 は，一般に公開されているデータセットを用いた3次元人体および手のメッシュ推定結果を示している．データセットでの3D人体および手のメッシュ推定結果を示す。 
我々の貢献は以下のようにまとめられる。  
* 我々は、1枚のRGB画像から人間の3Dポーズとメッシュを推定するための、新しい画像からリクセルへの予測ネットワークであるI2L-MeshNetを提案する。本システムは、入力画像の空間的関係を保持し、予測の不確実性をモデル化した、リクセルベースの1Dヒートマップを予測する。 
* この効率的なピクセルベースの1次元ヒートマップにより、本システムはヒートマップを十分な解像度で予測することができます。緻密なメッシュの頂点を特定するのに必要な十分な解像度でヒートマップを予測することができます。 
* 我々のI2L-MeshNetは、様々な3Dの人間のポーズとメッシュのデータセットにおいて、これまでの最先端の手法よりも優れていることを示している。


##  Conclusion
We propose a I2L-MeshNet, image-to-lixel prediction network for accurate 3D human pose and mesh estimation from a single RGB image.   
We convert the output of the network to the lixel-based 1D heatmap, which preserves the spatial relationship in the input image and models uncertainty of the prediction.   
Our lixel-based 1D heatmap requires much less GPU memory usage under the same heatmap resolution while producing better accuracy compared with a widely used voxel-based 3D heatmap.  
Our I2L-MeshNet outperforms previous 3D human pose and mesh estimation methods on various 3D human pose and mesh datasets.   
We hope our method can give useful insight to the following model-free
3D human pose and mesh estimation approaches.

我々は、1枚のRGB画像から正確な3D人間のポーズとメッシュを推定するために、I2L-MeshNet（画像からリクセルへの予測ネットワーク）を提案する。人間の姿勢とメッシュを1枚のRGB画像から正確に推定するI2L-MeshNetを提案します。  
このネットワークの出力をリクセルベースの1次元ヒートマップに変換することで、入力画像の空間的な関係を保持し、予測の不確実性をモデル化します。これにより、入力画像の空間的な関係を維持し、予測の不確実性をモデル化します。  
リクセルベースの1次元ヒートマップでは，同じヒートマップ解像度であれば，GPUのメモリ使用量が大幅に削減され，かつ精度も向上します．リクセルベースの1次元ヒートマップは，広く使われているボクセルベースの3次元ヒートマップよりも精度が高い。  
我々のI2L-MeshNetは、様々な3D人間のポーズとメッシュのデータセットにおいて、これまでの3D人間のポーズとメッシュの推定法よりも優れている。データセットにおいて、これまでの3D人間のポーズやメッシュ推定法よりも優れています。  
我々の手法が、以下のモデルフリーの3D人体姿勢・メッシュ推定手法に有用な知見を与えることを期待している。  
人間の姿勢とメッシュを推定する手法に役立つことを期待しています。

