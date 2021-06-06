## Title
I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image

I2L-MeshNet: 正確な3D人物像のための画像-画素間予測ネットワーク 1枚のRGB画像から人間の3Dポーズとメッシュを正確に推定する メッシュ推定

## Abstract.
Most of the previous image-based 3D human pose and mesh estimation methods estimate parameters of the human mesh model from an input image. However, directly regressing the parameters from the input image is a highly non-linear mapping because it breaks the spatial relationship between pixels in the input image. 
In addition, it cannot model the prediction uncertainty, which can make training harder. 
To resolve the above issues, we propose I2L-MeshNet, an image-to-lixel (line+pixel) prediction network. 
The proposed I2L-MeshNet predicts the per-lixel likelihood on 1D heatmaps for each mesh vertex coordinate instead of directly regressing the parameters.
Our lixel-based 1D heatmap preserves the spatial relationship in the input image and models the prediction uncertainty. We demonstrate the benefit of the image-to-lixel prediction and show that the proposed I2L-MeshNet outperforms previous methods. The code is publicly available 1.

これまでに開発された画像ベースの3次元人物姿勢・メッシュ推定手法の多くは 従来の画像ベースの3次元人物姿勢・メッシュ推定手法の多くは、入力画像から人物メッシュモデルのパラメータを推定するものでした。推定していました。しかし、入力画像からパラメータを直接回帰することは しかし、入力画像からパラメータを直接回帰することは、入力画像のピクセル間の空間的な関係を壊してしまうため、非常に非線形なマッピングとなります。
また，予測の不確実性をモデル化することができないため，学習が困難になる可能性があります．
上記の問題を解決するために、我々はI2L-MeshNetを提案する。(line+pixel) 予測ネットワークであるI2L-MeshNetを提案する．
提案されたI2L-MeshNetは、1つの画像上のリクセルごとの尤度を予測する。パラメータを直接回帰するのではなく，各メッシュ頂点座標の1次元ヒートマップ上でリクセル単位の尤度を予測する．
リクセルベースの1次元ヒートマップは リクセルベースの1Dヒートマップは、入力画像の空間的な関係を保持し、予測の不確実性をモデル化します。予測の不確実性をモデル化します。画像からリクセルへの予測の利点を実証し また，提案するI2L-MeshNetが従来の手法よりも優れていることを示した．コードは公開されています1。


## Introduction

3D human pose and mesh estimation aims to simultaneously recover 3D semantic human joint and 3D human mesh vertex locations. 
This is a very challenging task because of complicated human articulation and 2D-to-3D ambiguity. It can be used in many applications such as virtual/augmented reality and human action recognition.

人間の3次元ポーズとメッシュの推定は、人間の3次元的な関節と3次元的なメッシュの頂点位置を同時に復元することを目的としています。
これは、人間の複雑なアーティキュレーションや、2Dから3Dへの曖昧さのため、非常に困難なタスクです。これは、バーチャル/拡張現実や人間の行動認識など、多くのアプリケーションに利用できます。

SMPL [28] and MANO [42] are the most widely used parametric human body and hand mesh models respectively which can represent various human poses and identities. 
They produce 3D human joint and mesh coordinates from pose and identity parameters. 
Recent deep convolutional neural network (CNN)-based studies [20, 23, 40] for the 3D human pose and mesh estimation are based on the model-based approach, which trains a network to estimate SMPL/MANO parameters from an input image. 
On the other hand, there have been few methods based on model-free approach [9, 24], which estimates mesh vertex coordinates directly. 
They obtain the 3D pose by multiplying a joint regression matrix, included in the human mesh model, to the estimated mesh.

SMPL[28]とMANO[42]は，それぞれ最も広く使われているパラメトリックな人体と手のメッシュモデルで，様々な人間のポーズやアイデンティティを表現することができます．
これらのモデルは、ポーズとアイデンティティのパラメータから、3次元の人間の関節とメッシュの座標を生成します。
最近の深層畳み込みニューラルネットワーク（CNN）ベースの研究[20, 23, 40]は、モデルベースのアプローチに基づいており、入力画像からSMPL/MANOパラメータを推定するようにネットワークを学習させている。
その一方で，メッシュの頂点座標を直接推定するモデルフリー・アプローチに基づく手法もいくつか存在する[9, 24]．
これらの手法では，人間のメッシュモデルに含まれる関節回帰行列を，推定されたメッシュに乗じることで，3Dポーズを得ている．
