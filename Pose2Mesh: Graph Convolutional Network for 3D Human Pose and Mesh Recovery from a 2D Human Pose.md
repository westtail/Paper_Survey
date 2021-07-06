## Abstract. 概要 

Most of the recent deep learning-based 3D human pose and mesh estimation methods regress the pose and shape parameters of human mesh models, such as SMPL and MANO, from an input image.     
The first weakness of these methods is an appearance domain gap problem, due to different image appearance between train data from controlled environments, such as a laboratory, and test data from in-the-wild en- vironments.     
The second weakness is that the estimation of the pose parameters is quite challenging owing to the representation issues of 3D rotations.      
To overcome the above weaknesses, we propose Pose2Mesh, a novel graph convolutional neural network (GraphCNN)-based system that estimates the 3D coordinates of human mesh vertices directly from the 2D human pose.      
The 2D human pose as input provides essential human body articulation information, while having a relatively homogeneous geometric property between the two domains.     
Also, the proposed system avoids the representation issues, while fully exploiting the mesh topology using a GraphCNN in a coarse-to-fine manner.      
We show that our Pose2Mesh outperforms the previous 3D human pose and mesh estimation methods on various benchmark datasets.     
The codes are publicly available https://github.com/hongsukchoi/Pose2Mesh_RELEASE

最近の深層学習を用いた3D人間のポーズとメッシュの推定手法の多くは、SMPLやMANOなどの人間のメッシュモデルのポーズと形状のパラメータを入力画像から回帰させるものである。    
これらの手法の第一の弱点は、実験室のような制御された環境での訓練データと、野生の環境でのテストデータとの間で、画像の外観が異なることによる外観領域ギャップの問題です。    
第2の弱点は，3次元回転の表現問題のために，ポーズパラメータの推定が非常に困難であることである．     
上記の弱点を克服するために、我々は、人間の2次元ポーズから直接メッシュの頂点の3次元座標を推定する、グラフ畳み込みニューラルネットワーク（GraphCNN）ベースの新しいシステム、Pose2Meshを提案する。     
人間の2次元ポーズは、2つの領域の間で比較的均質な幾何学的特性を持ちながら、人体の重要な関節情報を提供します。    
また、提案システムでは、表現上の問題を回避しつつ、GraphCNNを用いてメッシュのトポロジーを粗いものから細かいものまで完全に利用しています。     
我々のPose2Meshは、様々なベンチマークデータセットにおいて、これまでの3D人間のポーズおよびメッシュ推定法を上回る性能を示した。    
コードは公開されています。https://github.com/hongsukchoi/Pose2Mesh_RELEASE

## Introduction イントロ

3D human pose and mesh estimation aims to recover 3D human joint and mesh vertex locations simultaneously.      
It is a challenging task due to the depth and scale ambiguity, and the complex human body and hand articulation.     
There have been diverse approaches to address this problem, and recently, deep learningbased methods have shown noticeable performance improvement.    

人間の3Dポーズとメッシュの推定は、人間の3D関節とメッシュの頂点の位置を同時に復元することを目的としています。     
これは、奥行きやスケールの曖昧さ、複雑な人体や手の関節の動きなどから、難しい課題となっています。    
この問題を解決するために、様々なアプローチが行われてきましたが、最近では、深層学習ベースの手法が顕著な性能向上を示しています。  

Most of the deep learning-based methods rely on human mesh models, such as SMPL [37] and MANO [54].      
They can be generally categorized into a modelbased approach and a model-free approach.       
The model-based approach trains a network to predict the model parameters and generates a human mesh by decoding them [5–7, 27, 31, 33, 47, 48, 52].      
On the contrary, the model-free approach regresses the coordinates of a 3D human mesh directly [15, 32].       
Both approaches compute the 3D human pose by multiplying the output mesh with a joint regression matrix, which is defined in the human mesh models [37, 54].     

深層学習を用いた手法の多くは，SMPL [37]やMANO [54]などの人体メッシュモデルに依存している．     
これらの手法は，一般に，モデルベースのアプローチとモデルフリーのアプローチに分類される．      
モデルベースのアプローチは，モデルのパラメータを予測するためにネットワークを学習し，それをデコードして人間のメッシュを生成するものである[5-7, 27, 31, 33, 47, 48, 52]．     
逆に，モデルフリー・アプローチは，3Dの人間メッシュの座標を直接回帰させるものである[15, 32]．      
どちらのアプローチも，出力されたメッシュに，人間のメッシュモデルで定義された関節回帰行列を乗じることで，人間の3次元ポーズを計算している[37, 54]．     

Although the recent deep learning-based methods have shown significant improvement, they have two major drawbacks.      
First, when tested on in-the-wild data, the methods inherently suffer from the appearance domain gap between controlled and in-the-wild environment data.     
The data captured from the controlled environments [22, 26] is valuable train data in 3D human pose and estimation, because it has accurate 3D annotations.      
However, due to the significant difference of image appearance between the two domains, such as backgrounds and clothes, an image-based approach cannot fully benefit from the data.      
The second drawback is that the pose parameters of the human mesh models might not be an appropriate regression target, as addressed in Kolotouros et al. [32].     
The SMPL pose parameters, for example, represent 3D rotations in an axisangle, which can suffer from the non-unique problem (i.e., periodicity).      
While many works [27, 33, 47] tried to avoid the periodicity by using a rotation matrix as the prediction target, it still has a non-minimal representation issue.      

最近の深層学習に基づく手法は大きな改善を見せているが、2つの大きな欠点がある。     
第一に、実世界のデータでテストした場合、制御された環境のデータと実世界の環境のデータとの間の外観領域のギャップに本質的に苦しんでいる。    
コントロールされた環境からキャプチャされたデータ[22, 26]は、正確な3Dアノテーションを持っているので、3Dの人間のポーズと推定における貴重なトレーニングデータです。     
しかし，背景や衣服など，2つの領域の間で画像の外観が大きく異なるため，画像ベースのアプローチではデータの恩恵を十分に受けることができません．     
2つ目の欠点は、Kolotourosら[32]が述べているように、人間のメッシュモデルのポーズパラメータが適切な回帰対象ではない可能性があることです。    
例えば，SMPL のポーズ・パラメータは，軸角での3次元回転を表しており，非一意性問題（周期性）が発生する可能性がある．     
多くの作品[27, 33, 47]では，回転行列を予測対象とすることで周期性を回避しようとしているが，それでも非最小表現の問題を抱えている．

To resolve the above issues, we propose Pose2Mesh, a graph convolutional system that recovers 3D human pose and mesh from the 2D human pose, in a model-free fashion.     
It has two advantages over existing methods.     
First, the proposed system benefits from a relatively homogeneous geometric property of the input 2D poses from controlled and in-the-wild environments.      
They not only alleviates the appearance domain gap issue, but also provide essential geometric information on the human articulation.      
Also, the 2D poses can be estimated accurately from in-the-wild images, since many well-performing methods [11, 44, 58, 66] are trained on large-scale in-the-wild 2D human pose datasets [2, 35].     
The second advantage is that Pose2Mesh avoids the representation issues of the pose parameters, while exploiting the human mesh topology (i.e., face and edge information).     
It directly regresses the 3D coordinates of mesh vertices using a graph convolutional neural network (GraphCNN) with graphs constructed from the mesh topology.

上記の問題を解決するために、我々は、モデルフリーで2Dの人間のポーズから3Dの人間のポーズとメッシュを復元するグラフ畳み込みシステム、Pose2Meshを提案する。    
これは既存の手法に比べて2つの利点がある。    
第一に、提案システムは、制御された環境と野生の環境からの入力2Dポーズの比較的均質な幾何学的特性から恩恵を受ける。     
これらは、外観領域のギャップの問題を軽減するだけでなく、人間のアーティキュレーションに関する重要な幾何学的情報を提供する。     
また、多くの優れた手法[11, 44, 58, 66]は、大規模な野生の2Dポーズデータセット[2, 35]を用いて学習されているため、野生の画像から2Dポーズを正確に推定することができる。    
第二の利点は，Pose2Mesh が，人間のメッシュのトポロジー（すなわち，顔とエッジの情報）を利用しながら，ポーズ・パラメータの表現上の問題を回避できることである．    
Pose2Meshは、メッシュのトポロジーから構築されたグラフを持つグラフ畳み込みニューラルネットワーク（GraphCNN）を用いて、メッシュの頂点の3D座標を直接回帰させる。

We designed Pose2Mesh in a cascaded architecture, which consists of PoseNet and MeshNet.    
PoseNet lifts the 2D human pose to the 3D human pose.     
MeshNet takes both 2D and 3D human poses to estimate the 3D human mesh in a coarseto-fine manner.      
During the forward propagation, the mesh features are initially processed in a coarse resolution and gradually upsampled to a fine resolution.     
Figure 1 depicts the overall pipeline of the system.

Pose2Meshは、PoseNetとMeshNetで構成されるカスケード・アーキテクチャで設計されています。   
PoseNetは、人間の2Dポーズを3Dポーズに変換します。    
MeshNetは、2Dと3Dの両方の人間のポーズを取り、3Dの人間のメッシュを粗くから細かくまで推定します。     
順方向の伝搬では，メッシュの特徴は最初は粗い解像度で処理され，徐々に細かい解像度にアップサンプリングされます。    
図1は、本システムの全体的なパイプラインを示しています。

The experimental results show that the proposed Pose2Mesh outperforms the previous state-of-the-art 3D human pose and mesh estimation methods [27,31,32] on various publicly available 3D human body and hand datasets [22, 39, 70].     
Particularly, our Pose2Mesh provides the state-of-the-art result on in-the-wild dataset [39], even when it is trained only on the controlled setting dataset [22].     
We summarize our contributions as follows

* We propose a novel system, Pose2Mesh, that recovers 3D human pose and mesh from the 2D human pose.     
The input 2D human pose lets Pose2Mesh robust to the appearance domain gap between controlled and in-the-wild environment data.    

* Our Pose2Mesh directly regresses 3D coordinates of a human mesh using GraphCNN.     
It avoids representation issues of the model parameters and leverages the pre-defined mesh topology.     

* We show that Pose2Mesh outperforms previous 3D human pose and mesh estimation methods on various publicly available datasets.   

実験結果によると，提案したPose2Meshは，公開されている様々な3D人体および手のデータセット[22, 39, 70]において，これまでの最先端の3D人体ポーズおよびメッシュ推定法[27,31,32]よりも優れた性能を発揮した．    
特に、我々のPose2Meshは、制御された設定データセット[22]のみで学習された場合でも、in-the-wildデータセット[39]で最先端の結果を提供している。    
我々の貢献を以下のように要約する。
* 人間の2Dポーズから人間の3Dポーズとメッシュを復元する新しいシステム「Pose2Mesh」を提案する。    
Pose2Meshは、人間の2Dポーズを入力とすることで、管理下のデータと野生環境のデータの間にある外見領域のギャップに頑健に対応することができる。   

* 当社のPose2Meshは、GraphCNNを用いて人間のメッシュの3次元座標を直接回帰させます。    
モデルパラメータの表現問題を回避し、事前に定義されたメッシュのトポロジーを活用しています。    
* Pose2Meshは、公開されている様々なデータセットにおいて、これまでの人間の3Dポーズやメッシュを推定する手法よりも優れた結果を示している。推定することができます。

## 2 Related works  関連作品

### 3D human body pose estimation. 人体の3Dポーズ推定。
Current 3D human body pose estimation methods can be categorized into two approaches according to the input type: an image-based approach and a 2D pose-based approach.     
The image-based approach takes an RGB image as an input for 3D body pose estimation.     
Sun et al. [59] proposed to use compositional loss, which exploits the joint connection structure.    
Sun et al. [60] employed soft-argmax operation to regress the 3D coordinates of body joints in a differentiable way.      
Sharma et al. [56] incorporated a generative model and depth ordering of joints to predict the most reliable 3D pose that corresponds to the estimated 2D pose.   

現在の3D人体姿勢推定法は、入力タイプによって、画像ベースのアプローチと2Dポーズベースのアプローチの2つに分類されます。   
画像ベースのアプローチでは、RGB画像を3Dボディ・ポーズ推定の入力として使用する。    
Sunら[59]は，関節の接続構造を利用したコンポジション・ロスを提案している．     
Sunら[60]は、体の関節の3次元座標を微分可能な方法で回帰させるために、soft-argmax演算を採用した。     
Sharmaら[56]は、生成モデルと関節の深さ方向の順序付けを用いて、推定された2Dポーズに対応する最も信頼性の高い3Dポーズを予測した。    

The 2D pose-based approach lifts the 2D human pose to the 3D space.      
Martinez et al. [40] introduced a simple network that consists of consecutive fullyconnected layers, which lifts the 2D human pose to the 3D space.       
Zhao et al. [68] developed a semantic GraphCNN to use spatial relationships between joint coordinates.      
Our work follows the 2D pose-based approach, to make the Pose2Mesh more robust to the domain difference between the controlled environment of the training set and in-the-wild environment of the testing set.    

2D ポーズベースのアプローチは、人間の 2D ポーズを 3D 空間に持ち上げるものである。     
Martinezら[40]は、完全に接続された連続した層からなる単純なネットワークを導入し、2Dの人間のポーズを3D空間に持ち上げることに成功した。      
Zhaoら[68]は、関節座標間の空間的関係を利用するために、セマンティックGraphCNNを開発した。     
本研究では、2Dポーズベースのアプローチを採用し、トレーニングセットの制御された環境とテストセットの野生環境との間の領域差に対してPose2Meshをよりロバストにする。    

### 3D human body and hand pose and mesh estimation. 人体と手の3Dポーズとメッシュの推定。
A model-based approach trains a neural network to estimate the human mesh model parameters [37, 54].      
It has been widely used for the 3D human mesh estimation, since it does not necessarily require 3D annotation for mesh supervision.       
Pavlakos et al. [52] proposed a system that could be only supervised by 2D joint coordinates and silhouette.      
Omran et al. [47] trained a network with 2D joint coordinates, which takes human part segmentation as input.      
Kanazawa et al. [27] utilized adversarial loss to regress plausible SMPL parameters.      
Baek et al. [5] trained a CNN to estimate parameters of the MANO model using neural renderer [29].      
Kolotouros et al. [31] introduced a self-improving system that consists of SMPL parameter regressor and iterative fitting framework [6].    

モデルベース・アプローチは，ニューラル・ネットワークを訓練して，人間のメッシュ・モデル・パラメータを推定するものである[37, 54]．      
これは，メッシュを監視するための3次元アノテーションを必ずしも必要としないため，人間の3次元メッシュ推定に広く利用されている．      
Pavlakosら[52]は，2次元の関節座標とシルエットのみでスーパーバイズできるシステムを提案した．     
Omranら[47]は，2次元の関節座標を用いてネットワークを学習し，人間の部位のセグメンテーションを入力としている．     
Kanazawaら[27]は，SMPLのもっともらしいパラメータを回帰するために逆問題を利用した．     
Baekら[5]は、ニューラルレンダラー[29]を用いて、MANOモデルのパラメータを推定するためにCNNを学習しました。     
Kolotourosら[31]は，SMPLパラメータの回帰器と反復的なフィッティングフレームワークからなる自己改善システムを紹介した[6]．   

Recently, the advance of fitting frameworks [6, 50] has motivated a modelfree approach, which estimates human mesh coordinates directly.      
It enabled researchers to obtain 3D mesh annotation, which is essential for the model-free methods, from in-the-wild data.     
Kolotouros et al. [32] proposed a GraphCNN, which learns the deformation of the template body mesh to the target body mesh.      
Ge et al. [15] adopted a GraphCNN to estimate vertices of hand mesh. Moon et al. [45] proposed a new heatmap representation, called lixel, to recover 3D human meshes.  

最近では，フィッティングフレームワーク[6, 50]の進歩により，人間のメッシュ座標を直接推定するモデルフリーアプローチが提唱されています．     
これにより，モデルフリー手法に不可欠な3次元メッシュアノテーションを，実世界のデータから得ることができるようになった．    
Kolotourosら[32]は，テンプレートボディメッシュからターゲットボディメッシュへの変形を学習するGraphCNNを提案した．     
Geら[15]はGraphCNNを用いて手のメッシュの頂点を推定している。Moonら[45]は，人間の3次元メッシュを復元するために，lixelと呼ばれる新しいヒートマップ表現を提案した． 

Our Pose2Mesh differs from the above methods, which are image-based, in that it uses the 2D human pose as an input.     
The proposed system can benefit from the data with 3D annotations, which are captured from controlled environments [22, 26], without the appearance domain gap issue.

我々のPose2Meshは，画像ベースの上記の手法とは異なり 人間の2Dポーズを入力として使用している点である。    
提案されたシステムは 提案システムは、制御された環境からキャプチャされた3Dアノテーションを持つデータ[22, 26]から、外観ドメインギャップの問題なく恩恵を受けることができる。

### GraphCNN for mesh processing. GraphCNNによるメッシュ処理。
Recently, many methods consider a mesh as a graph structure and process it using the GraphCNN, since it can fully exploit mesh topology compared with simple stacked fully-connected layers.       
Wang et al. [65] adopted a GraphCNN to learn a deformation from an initial ellipsoid mesh to the target object mesh in a coarse-to-fine manner.       
Verma et al. [63] proposed a novel graph convolution operator for the shape correspondence problem.      
Ranjan et al. [53] also proposed a GraphCNN-based VAE, which learns a latent space of the human face meshes in a hierarchical manner.      

近年、メッシュをグラフ構造とみなし、GraphCNNを用いてメッシュを処理する手法が多くなってきている。GraphCNNは、単純に積み上げた完全連結層に比べて、メッシュのトポロジーを十分に活用できるからである。      
Wangら[65]は、GraphCNNを用いて、楕円体の初期メッシュから対象物のメッシュへの変形を、粗いものから細かいものへと学習している。      
Vermaら[63]は、形状対応問題に対する新しいグラフ畳み込み演算子を提案した。     
Ranjanら[53]もGraphCNNをベースにしたVAEを提案しており，人間の顔のメッシュの潜在空間を階層的に学習することに成功している．     
