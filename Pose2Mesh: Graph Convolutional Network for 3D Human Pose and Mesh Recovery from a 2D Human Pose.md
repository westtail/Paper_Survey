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

## 3 PoseNet ポーズネット
### 3.1 Synthesizing errors on the input 2D pose 入力2Dポーズの合成誤差

PoseNet estimates the root joint-relative 3D pose P3D∈RJ×3 from the 2D pose, where J denotes the number of human joints.      
We define the root joint of the human body and hand as pelvis and wrist, respectively.      
The estimated 2D pose often contains errors [55], especially under severe occlusions or challenging poses.       
To make PoseNet robust to the errors, we synthesize 2D input poses by adding realistic errors on the ground truth 2D pose, following [10, 44], during the training stage.       
We represent the estimated 2D pose or the synthesized 2D pose as P2D ∈ R J×2.     

PoseNetは、2Dポーズからルートジョイント相対的な3DポーズP3D∈RJ×3を推定する（Jは人間の関節数を表す）。     
ここでは、人体と手の根元の関節をそれぞれ骨盤と手首と定義する。     
推定された2次元ポーズには，特に厳しいオクルージョンや挑戦的なポーズの下では，しばしば誤差が含まれる[55]．      
PoseNetが誤差に対してロバストになるように、学習段階において、[10, 44]に従い、グランドトゥルースの2Dポーズに現実的な誤差を加えて、2D入力ポーズを合成する。      
推定された2Dポーズまたは合成された2Dポーズを、P2D∈R J×2として表現する。

### 3.2 2D input pose normalization 2D入力ポーズの正規化
We apply standard normalization to P2D, following [10,64].     
For this, we subtract the mean from P2D and divide it by the standard deviation, which becomes P¯ 2D.      
The mean and the standard deviation of P2D represent the 2D location and scale of the subject, respectively.     
This normalization is necessary because P3D is independent of scale and location of the 2D input pose P2D.     

P2Dには、[10,64]に準拠した標準的な正規化を適用します。    
これには、P2Dから平均を差し引き、標準偏差で割ると、P¯ 2Dとなります。     
P2Dの平均と標準偏差は、それぞれ被写体の2D位置とスケールを表しています。    
P3Dは、2D入力ポーズP2Dのスケールと位置に依存しないため、このような正規化が必要となる。    

### 3.3 Network architecture ネットワーク・アーキテクチャ
The architecture of the PoseNet is based on that of [10, 40]. The normalized 2D input pose P¯ 2D is converted to a 4096 dimensional feature vector through a fully-connected layer.       
Then, it is fed to the two residual blocks [21].      
Finally, the output feature vector of the residual blocks is converted to (3J)-dimensional vector, which represents P3D, by a full-connected layer.    

PoseNetのアーキテクチャは，[10, 40]のものに基づいています。正規化された 2D 入力ポーズ P¯ 2D は，完全連結層を介して 4096 次元の特徴ベクトルに変換される．      
その後、2つの残差ブロック[21]に供給される。     
最後に、残差ブロックの出力特徴ベクトルは、完全連結層によって、P3Dを表す(3J)次元のベクトルに変換される。   

### 3.4 Loss function 損失関数
We train the PoseNet by minimizing L1 distance between the predicted 3D pose P3D and groundtruth.     
The loss function Lpose is defined as follows:     
Lpose = || P 3D − P 3D∗ ||   (1)     
where the asterisk indicates the groundtruth.     

予測された3DポーズP3Dとグランドトゥルースの間のL1距離を最小化することで、PoseNetを学習する。    
損失関数Lposeは以下のように定義される。    
Lpose = || P 3D - P 3D∗ || (1)     
ここで，アスタリスクはグランドトゥルースを示す。  

### meshnet image
Fig. 2: The coarsening initially generates multiple coarse graphs from GM, and adds fake nodes without edges to each graph, following [13].     
The numbers of vertices range from 96 to 12288 for body meshes and from 68 to 1088 for hand meshes.      
Fig. 3: The network architecture of MeshNet

図2：粗視化では，最初にGMから複数の粗いグラフを生成し，[13]に倣って各グラフに辺のない偽のノードを追加する．     
頂点の数は，体のメッシュでは96から12288まで，手のメッシュでは68から1088までとなっている．    
図3：MeshNetのネットワークアーキテクチャ

##  MeshNet

### 4.1 Graph convolution on pose ポーズに対するグラフの畳み込み
MeshNet concatenates P¯ 2D and P3D into P∈RJ×5.     
Then, it estimates the root joint-relative 3D mesh M∈RV×3 from P, where V denotes the number of human mesh vertices.      
To this end, MeshNet uses the spectral graph convolution [8, 57], which can be defined as the multiplication of a signal x∈RN with afilter gθ = diag(θ) in Fourier domain as follows:     
gθ ∗ x = U gθUT x, (2)     
where graph Fourier basis U is the matrix of the eigenvectors of the normalized graph Laplacian L [12], and U T x denotes the graph Fourier transform of x.    
Specifically, to reduce the computational complexity, we design MeshNet to be based on Chebysev spectral graph convolution [13].   

MeshNetは、P¯ 2DとP3DをP∈RJ×5に連結します。    
ここで、Vは人間のメッシュの頂点の数を表します。     
このために，MeshNetはスペクトルグラフコンボリューション[8, 57]を用いる．これは，信号x∈RNとフーリエ領域のafilter gθ = diag(θ)との乗算として，以下のように定義できる。    
gθ ∗ x = U gθUT x, (2)     
ここで，グラフフーリエ基底Uは，正規化グラフラプラシアンLの固有ベクトルの行列である[12]。また，U T xはxのグラフフーリエ変換を表す。   
具体的には，計算量を減らすために，MeshNetはチェビセブ・スペクトル・グラフ・コンボリューション[13]に基づいて設計されている．  

# ここから5章まではあとで訳します


## 5 Implementation Details 実装の詳細
PyTorch [49] is used for implementation. We first pre-train our PoseNet, and then train the whole network, Pose2Mesh, in an end-to-end manner.       
Empirically, our two-step training strategy gives better performance than the one-step training.     
The weights are updated by the Rmsprop optimization [61] with a mini-batch size of 64.     
We pre-train PoseNet 60 epochs with a learning rate 10−3.     
The learningrate is reduced by a factor of 10 after the 30th epoch.       
After integrating the pretrained PoseNet to Pose2Mesh, we train the whole network 15 epochs with a learning rate 10−3.      
The learning rate is reduced by a factor of 10 after the 12th epoch. In addition, we set λe to 0 until 7 epoch on the second training stage, since it tends to cause local optima at the early training phase.      
We used four NVIDIA RTX 2080 Ti GPUs for Pose2Mesh training, which took at least a half day and at most two and a half days, depending on the training datasets.     
In inference time, we use 2D pose outputs from Sun et al. [58] and Xiao et al. [66].       
They run at 5 fps and 67 fps respectively, and our Pose2Mesh runs at 37 fps.           
Thus, the proposed system can process from 4 fps to 22 fps in practice, which shows the applicability to real-time applications.  

実装にはPyTorch[49]を使用しています。まず，PoseNet を事前に学習し，次にネットワーク全体である Pose2Mesh をエンド・ツー・エンドで学習します．      
経験的に、我々の2段階のトレーニング戦略は、1段階のトレーニングよりも優れた性能を発揮する。    
重みはRmsprop最適化[61]によって更新され，ミニバッチサイズは64となっています．    
学習率10-3で，PoseNetを60エポックで事前学習します．    
30エポック以降は，学習率を10分の1に下げています．      
事前学習されたPoseNetをPose2Meshに統合した後，学習率10-3で15エポックのネットワーク全体の学習を行います．     
12回目のエポック以降は，学習率を10分の1に下げます．また，λeは初期の学習段階で局所最適を起こしやすいため，2回目の学習段階で7エポックまでは0に設定した．     
Pose2Meshの学習には，NVIDIA RTX 2080 Ti GPUを4台使用し，学習データセットに応じて最低でも半日，最大で2日半を要した．    
推論には，Sunら[58]およびXiaoら[66]の2Dポーズ出力を使用した．      
これらはそれぞれ5 fpsと67 fpsで動作し、我々のPose2Meshは37 fpsで動作する。          
このように，提案システムは実際には4fpsから22fpsの処理が可能であり，リアルタイムアプリケーションへの適用が可能であることがわかる．  

## 6 Experiment 実験
### 6.1 Datasets and evaluation metrics データセットと評価指標
#### Human3.6M.    
Human3.6M [22] is a large-scale indoor 3D body pose benchmark, which consists of 3.6M video frames.    
The groundtruth 3D poses are obtained using a motion capture system, but there are no groundtruth 3D meshes.      
As a result, for 3D mesh supervision, most of the previous 3D pose and mesh estimation works [27, 31, 32] used pseudo-groundtruth obtained from Mosh [36].       
However, due to the license issue, the pseudo-groundtruth from Mosh is not currently publicly accessible.       
Thus, we generate new pseudo-groundtruth 3D meshes by fitting SMPL parameters to the 3D groundtruth poses using SMPLify-X [50].       
For the fair comparison, we trained and tested previous state-of-the-art methods on the obtained groundtruth using their officially released code.       
Following [27, 51], all methods are trained on 5 subjects (S1, S5, S6, S7, S8) and tested on 2 subjects (S9, S11).   
We report our performance for the 3D pose using two evaluation metrics.       
One is mean per joint position error (MPJPE) [22], which measures the Euclidean distance in millimeters between the estimated and groundtruth joint coordinates,after aligning the root joint.      
The other one is PA-MPJPE, which calculates MPJPE after further alignment (i.e., Procrustes analysis (PA) [17]).       
J M is used for the estimated joint coordinates. We only evaluate 14 joints out of 17 estimated joints following [27, 31, 32, 52].

Human3.6M [22]は，360万枚のビデオフレームからなる，大規模な屋内3Dボディポーズベンチマークである．   
このベンチマークでは，モーションキャプチャシステ ムを用いて 3D ポーズを取得しているが，3D メッシュは取得していない．      
そのため，3Dメッシュの監視には，これまでの3Dポーズやメッシュ推定の作品[27, 31, 32]のほとんどが，Mosh[36]から得られた擬似的なグランドトゥルースを使用していました．      
しかし，ライセンスの問題から，Moshからの擬似地表面情報は現在のところ公開されていない．      
そこで，SMPLify-X [50]を用いて，SMPLのパラメータを3D groundtruthのポーズにフィットさせることで，新たな疑似groundtruthの3Dメッシュを生成した．      
公平に比較するために，公式にリリースされているコードを用いて，得られたグランドトゥルースに対して過去の最先端の手法をトレーニングし，テストした．      
[27, 51] に従って，すべての手法は，5人の被験者（S1, S5, S6, S7, S8）でトレーニングされ，2人の被験者（S9, S11）でテストされた．     
ここでは、2つの評価指標を用いて、3Dポーズの性能を報告します。      
一つは，MPJPE（mean per joint position error）[22]であり，根元の関節をアライメントした後の推定関節座標と実測関節座標の間のユークリッド距離をミリメートル単位で測定する．     
もう一つはPA-MPJPEであり，これはMPJPEをさらにアライメント（すなわちProcrustes analysis (PA) [17]）した後に計算するものである．      
J Mは，推定された関節座標に使用される．我々は、[27, 31, 32, 52]に従い、17個の推定関節のうち、14個の関節のみを評価する。

#### 3DPW. 
3DPW [39] is captured from in-the-wild and contains 3D body pose and mesh annotations. It consists of 51K video frames, and IMU sensors are leveraged to acquire the groundtruth 3D pose and mesh.     
We only use the test set of 3DPW for evaluation following [31].      
MPJPE and mean per vertex position error (MPVPE) are used for evaluation.      
14 joints from J M, whose joint set follows that of Human3.6M, are evaluated for MPJPE as above.      
MPVPE measures the Euclidean distance in millimeters between the estimated and groundtruth vertex coordinates, after aligning the root joint.      

3DPW [39]は実社会で撮影されたもので、3Dのボディ・ポーズとメッシュ・アノテーションが含まれています。メッシュのアノテーションが含まれています．51Kのビデオフレームで構成されており、IMUセンサーを利用して 3Dポーズとメッシュを取得しています。    
評価に使用したのは 3DPWのテストセットのみを使用して評価しています。     
MPJPEおよびMean per vertex position error (MPVPE) MPVPE）を用いて評価している．     
関節セットがHuman3.6Mのものと同じであるJ Mの14関節 の14関節について，上記と同様にMPJPEの評価を行った．     
MPVPEは MPVPEは，関節の根元の位置を合わせた後，推定された頂点座標とグランドトゥルースの頂点座標の間のユークリッド距離をミリ単位で測定する．のユークリッド距離を測定します。 

#### COCO. 
COCO [35] is an in-the-wild dataset with various 2D annotations such as detection and human joints.     
To exploit this dataset on 3D mesh learning, Kolotouros et al. [31] fitted SMPL parameters to 2D joints using SMPLify [6].     
Following them, we use the processed data for training.       

COCO[35]は、様々な2Dアノテーションを持つ野生下のデータセットです。検出や人間の関節など、様々な2Dアノテーションが含まれている。    
このデータセットを3次元メッシュ学習に活用するために、Kolotouros et al, Kolotourosら[31]は、SMPLify[6]を用いてSMPLパラメータを2D関節に適合させた。    
彼らに倣って、我々もこの処理済みのデータを学習に使用する。    

#### MuCo-3DHP.      
MuCo-3DHP [42] is synthesized from the existing MPI-INF3DHP 3D single-person pose estimation dataset [41].      
It consists of 200K frames, and half of them have augmented backgrounds.      
For the background augmentation, we use images of COCO that do not include humans to follow Moon et al. [43].      
Following them, we use this dataset only for the training.     

MuCo-3DHP [42]は，既存のMPI-INF3DHP 3D single-person pose estimation dataset [41]から合成したものである．     
このデータセットは200Kフレームで構成されており，そのうちの半分は背景が拡張されています．     
背景の拡張には，Moonら[43]に倣い，人間を含まないCOCOの画像を使用している．     
彼らに倣って、このデータセットはトレーニングにのみ使用している。    

FreiHAND. 
FreiHAND [70] is a large-scale 3D hand pose and mesh dataset.      
It consists of a total of 134K frames for training and testing.       
Following Zimmermann et al. [70], we report PA-MPVPE, F-scores, and additionally PA-MPJPE of Pose2Mesh.     
J M is evaluated for the joint errors.     

FreiHAND[70]は、大規模な3Dハンドポーズとメッシュのデータセットである。     
学習用とテスト用の合計134Kフレームから構成されている。      
Zimmermannら[70]に従い，PA-MPVPE，F-scores，さらにPose2MeshのPA-MPJPEを報告する．    
J Mはジョイントエラーで評価しています。    

#　実験は今後和訳

7 Discussion 議論
Although the proposed system benefits from the homogeneous geometric property of input 2D poses from different domains, it could be challenging to recover various 3D shapes solely from the pose.       
While it may be true, we found that the 2D pose still carries necessary information to reason the corresponding 3D shape.       
In the literature, SMPLify [6] has experimentally verified that under the canonical body pose, utilizing 2D pose significantly drops the body shape fitting error compared to using the mean body shape.       
We show that Pose2Mesh can recover various body shapes from the 2D pose in the supplementary material.   

提案されたシステムは、異なるドメインからの入力2Dポーズの均質な幾何学的特性の恩恵を受けているが、ポーズのみから様々な3D形状を復元するのは難しいかもしれない。      
しかし、我々は、2Dポーズが、対応する3D形状を推論するために必要な情報を含んでいることを発見した。      
文献では、SMPLify [6]が、カノニカルなボディポーズの下で、2Dポーズを利用すると、平均ボディシェイプを利用する場合に比べて、ボディシェイプのフィッティングエラーが大幅に減少することを実験的に検証しています。      
Pose2Meshでは、2Dポーズから様々な体形を復元できることを補足資料で示しています。    

## 8 Conclusion おわりに
We propose a novel and general system, Pose2Mesh, for 3D human mesh and pose estimation from a 2D human pose.      
The input 2D pose enables the system to benefit from the 3D data captured from the controlled settings without the appearance domain gap issue.      
The model-free approach using GraphCNN allows it to fully exploit mesh topology, while avoiding the representation issues of the 3D rotation parameters.      
We plan to enhance the shape recover capability of Pose2Mesh using denser keypoints or part segmentation, while maintaining the above advantages.    

我々は、人間の2Dポーズから人間の3Dメッシュとポーズを推定するための新規かつ汎用的なシステム、Pose2Meshを提案する。     
Pose2Mesh は、2D ポーズを入力することで、外見上のドメインギャップの問題なしに、制御された設定からキャプチャされた 3D データの恩恵を受けることができます。     
GraphCNNを用いたモデルフリーのアプローチにより、メッシュのトポロジーを十分に活用しつつ、3D回転パラメータの表現上の問題を回避することができます。3D回転パラメータの表現問題を回避することができます。     
今後は、上記の利点を維持しつつ、キーポイントの高密度化やパーツのセグメンテーションにより、Pose2Meshの形状復元能力を向上させていく予定です。   
