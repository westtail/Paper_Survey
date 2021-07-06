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
