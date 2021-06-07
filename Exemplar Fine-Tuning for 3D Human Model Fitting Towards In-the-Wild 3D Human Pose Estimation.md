# title 
Exemplar Fine-Tuning for 3D Human Model Fitting Towards In-the-Wild 3D Human Pose Estimation  
3次元人体モデルのフィッティングのための模範的な微調整と、実世界での3次元人体ポーズの推定



# Abstract
We present Exemplar Fine-Tuning (EFT), a new method to fit a 3D parametric human model to a single RGB input image cropped around a person with 2D keypoint annotations.    
While existing parametric human model fitting approaches, such as SMPLify, rely on the “view-agnostic” human pose priors to enforce the output in a plausible 3D pose space, EFT exploits the pose prior that comes from the specific 2D input observations by leveraging a fully-trained 3D pose regressor.   
We thoroughly compare our EFT with SMPLify, and demonstrate that EFT produces more reliable and accurate 3D human fitting outputs on the same inputs.     
Especially, we use our EFT to augment a large scale in-the-wild 2D keypoint datasets, such as COCO and MPII, with plausible and convincing 3D pose fitting outputs.    
We demonstrate that the pseudo ground-truth 3D pose data by EFT can supervise a strong 3D pose estimator that outperforms the previous state-of-the-art in the standard outdoor benchmark (3DPW), even without using any ground-truth 3D human pose datasets such as Human3.6M.    
Our code and data are available at https://github.com/facebookresearch/eft.

本研究では、2次元のキーポイントアノテーションを持つ人物の周囲を切り取ったRGBの入力画像に、3次元のパラメトリック人体モデルを適合させる新しい手法であるExemplar Fine-Tuning（EFT）を発表します。   
SMPLifyなどの既存のパラメトリック人体モデルフィッティング手法は、3Dポーズ空間での出力を保証するために、「視界にとらわれない」人間のポーズプリオールに依存しているが、EFTは、完全に訓練された3Dポーズレグレッサーを活用することで、特定の2D入力観測から得られるポーズプリフォームを利用する。  
我々のEFTとSMPLifyを徹底的に比較し、EFTが同じ入力に対して、より信頼性の高い、正確な3Dフィッティング出力を生成することを実証しました。    
特に、我々のEFTを用いて、COCOやMPIIなどの大規模なin-the-wildの2Dキーポイントデータセットを、もっともらしく説得力のある3Dポーズフィッティング出力で補強しました。   
EFTによって擬似的に得られた3Dポーズデータは、Human3.6Mのような人間の3Dポーズデータを使わなくても、標準的な屋外ベンチマーク（3DPW）において、従来の最先端技術を凌駕する強力な3Dポーズ推定量をスーパーバイズできることを実証した。   
我々のコードとデータは、https://github.com/facebookresearch/eft。



# Introduction
We consider the problem of reconstructing the pose of humans in 3D from single 2D images, a key task in applications such as human action recognition, human-machine interaction and virtual and augmented reality.    
Since individual 2D images do not contain sufficient information for 3D reconstruction, algorithms must supplement the missing information by a learned prior on the plausible 3D poses of the human body.   
Established approaches such as SMPLify [1, 2] cast this as fitting the parameters of a 3D human model [3, 4, 5] to the location of 2D keypoints, manually or automatically annotated in images.    
They regularize the solution by means of a “view-agnostic” 3D pose prior, which incurs some limitations: fitting ignores the RGB images themselves, the 3D priors, often learned separately in laboratory conditions, lack realism, and balancing between the prior and the data term (e.g., 2D keypoint error) is difficult.

この問題は、人間の行動認識、ヒューマンマシンインタラクション、バーチャルリアリティや拡張現実などのアプリケーションにおいて重要な課題です。   
個々の2D画像には、3D再構成に必要な十分な情報が含まれていないため、アルゴリズムは、人体の妥当な3Dポーズに関する学習済みの事前情報によって、不足している情報を補う必要がある。  
SMPLify [1, 2]などの確立されたアプローチでは，3D人体モデル [3, 4, 5]のパラメータを，画像に手動または自動で注釈された2Dキーポイントの位置に適合させることで，これを実現しています．   
これらの手法では，「視界に依存しない」3Dポーズ事前分布を用いて解を正則化しているが，これにはいくつかの限界がある．すなわち，フィッティングではRGB画像そのものが無視されること，3D事前分布は実験室環境で個別に学習されることが多いためリアリティに欠けること，事前分布とデータ項（2Dキーポイントの誤差など）のバランスをとるのが難しいこと，などである．

In this paper, we present Exemplar Fine-Tuning (EFT), a new technique to fit 3D parametric models to 2D annotations which overcomes these limitations and results in much better 3D reconstructions.   
The idea of EFT is to start from an image-based 3D pose regressor such as [6, 7, 8]. The pretrained regressor implicitly embodies a strong pose prior conditional on the observation of the RGB input image, providing a function Θ = Φw(I) sending the image I to the parameters Θ of the 3D body.      
Our intuition is to look at this function as a conditional re-parameterization of pose,　where the conditioning factor is the image I, and the new parameters are the weights w of the underlying (neural network) regressor.    
We show that, when this re-parameterization is used, fitting 2D keypoints results in better 3D reconstructions than using existing fitting methods such as SMPLify, while also improving the output of the regressor Φ itself.   

本論文では、3Dパラメトリックモデルを2Dアノテーションに適合させる新しい技術であるEFT（Exemplar Fine-Tuning）を紹介します。  
EFTのアイデアは、[6, 7, 8]のような画像ベースの3Dポーズレグレッサーから始めることである。前もって学習されたレグレッサーは、RGB入力画像の観察を条件とした強力なポーズの事前情報を暗黙のうちに具現化し、画像Iを3DボディのパラメータΘに送る関数Θ=Φw(I)を提供するものである。     
我々の直感では、この関数をポーズの条件付き再パラメータ化と見なし、条件付け要因を画像Iとし、新しいパラメータを基礎となる（ニューラルネットワーク）回帰器の重みwとしています。   
我々は，この再パラメータ化を用いた場合，2Dキーポイントのフィッティングは，SMPLifyなどの既存のフィッティング手法を用いた場合よりも優れた3D再構成となり，同時に，リグレッサーΦ自体の出力も改善されることを示す．

The name of our technique is justified by the fact that fitting the parameters w of the predictor to a 2D example is similar to performing a training step for the regressor Φ.  
However, this is only done for the purpose of reconstructing a particular exemplar, not for learning the network. In fact, the updated parameters are discarded before processing the next sample.

我々の手法の名前は、予測器のパラメータwを2Dの例にフィットさせることは、リグレッサΦのトレーニングステップを実行することに似ているという事実によって正当化される。  
しかし、これは特定の模範例を再構築する目的でのみ行われ、ネットワークの学習のためではない。実際、更新されたパラメータは、次のサンプルを処理する前に破棄される。  

Figure 1: Exemplar Fine-Tuning (EFT) fits a parametric 3D human model given as input a single RGB image and 2D keypoint annotations.  
(Column 1) Input image with 2D keypoint annotations;   
(Columns 2-4) EFT iterations;    
(Column 5) a side view;   
(Column 6-7) output of SMPLify [1] without using a 3D pose priors;   
(Column 8-9) SMPLify [1] with 3D pose priors.   
For SMPLify, a staged technique is used to avoid local minima (100 iterations in total), while EFT optimizes all parts together without using any external 3D pose priors.

Figure 1: Exemplar Fine-Tuning (EFT) は，1枚のRGB画像と2次元のキーポイントのアノテーションを入力として，パラメトリックな3次元人体モデルにフィットする． 
(列1) 入力画像と2Dキーポイントのアノテーション．  
(コラム2-4) EFTのイテレーション。   
(5列目)横から見た図。  
(6-7列目) 3Dポーズプリオールを使用しないSMPLify [1]の出力。  
(8-9列目) 3Dポーズプリオールを用いたSMPLify [1]の出力。  
SMPLifyでは、局所的な最小値を避けるために段階的な手法が用いられています（合計100回の反復）。EFTでは、外部の3Dポーズ・プライアを使用せずに、すべてのパーツをまとめて最適化しています。

We show two important application of this technique.   
The first is ‘direct’: for in-the-wild data, EFT results in better single-view 3D body pose reconstruction than any existing approach [9, 6, 10, 11, 12, 13, 14, 7, 15, 16, 17], and thus can be used as a drop-in replacement of these techniques in current applications.   
As shown in Fig. 1, EFT fits are particularly robust to challenging 2D poses than defy traditional fitting and regression methods [3, 5, 4, 1, 2].   
This is possible because EFT can leverage the implicit prior learned by the neural network regressor.   
Furthermore, this prior is conditioned on the specific the RGB input image, which contains more information than the 2D keypoint locations.   
At the same time, it results in more accurate 2D fits than the regressor alone.

この技術の重要な応用例を2つ紹介します。  
1つ目は「直接的」なものです。EFTは、実世界のデータに対して、既存のどの手法よりも優れたシングルビューの3Dボディポーズ再構成を行うことができ[9, 6, 10, 11, 12, 13, 14, 7, 15, 16, 17]、したがって、現在のアプリケーションにおいて、これらの手法の代わりにドロップインとして使用することができます。  
図1に示すように，EFTによるフィッティングは，従来のフィッティング手法や回帰手法では対応できないような難しい2Dポーズに対しても，特にロバストに対応しています[3, 5, 4, 1, 2]。  
これは、EFTがニューラルネットワークの回帰器によって学習された暗黙の事前情報を活用できるためです。  
さらに，この事前情報には 2次元のキーポイントの位置よりも多くの情報を含むRGBの入力画像を条件としています。  
同時に、レグレッサーのみの場合よりも、より正確な2Dフィットが得られます。

The second application of EFT is to generate 3D body pose data in the wild.    
We show this  by taking an existing large-scale 2D dataset such as COCO [18] and using EFT to augment it with approximate 3D body pose annotations.     
Remarkably, we show that existing supervised pose regressor methods can be trained using these pseudo-3D annotations as well or better than using ground truth 3D annotations in existing 3D datasets [19, 20, 21], which, for the most part, are collected in laboratory condition.    
In fact, we show that our 3D-fied in the wild data, which we will release to the public, is sufficient to train state-of-the-art 3D pose regressors by itself, outperforming methods trained on the combination of datasets with 3D and 2D ground-truth [7, 22, 8].   

EFTの2つ目の用途は、自然の中で3Dボディポーズデータを生成することです。   
COCO[18]のような既存の大規模な2DデータセットにEFTを用いて、3Dボディポーズの近似アノテーションを追加することで、このことを示しています。    
驚くべきことに、既存の教師付きポーズ回帰法は、この擬似3Dアノテーションを用いて、実験室で収集された既存の3Dデータセット[19, 20, 21]のグランドトゥルース3Dアノテーションと同等以上に学習できることを示している。   
実際，我々が公開している3D-fied in the wildデータは，最先端の3Dポーズリグレッサを単独で学習するのに十分であり，3Dと2Dのグランドトゥルースを持つデータセットの組み合わせで学習した手法を上回ることを示している[7, 22, 8]．  




#  Related Work 
Deep learning has significantly advanced 2D pose recognition [23, 24, 25, 26, 27, 28], facilitating the more challenging task of 3D reconstruction [9, 6, 10, 11, 12, 13, 14, 7, 15, 17, 16, 22], our focus.

ディープラーニングは、2Dポーズ認識を大幅に進化させ[23, 24, 25, 26, 27, 28]、我々が注目する3D再構築というより困難なタスクを容易にしている[9, 6, 10, 11, 12, 13, 14, 7, 15, 17, 16, 22]。

Single-image 3D human pose estimation.   
Single-view 3D pose reconstruction methods differ in how they incorporate a 3D pose prior and in how they perform the prediction.    
Fitting-based methods assume a 3D body model such as SMPL [3] and SCAPE [29], and use an optimization algorithm to fit it to the 2D observations.    
While early approaches [30, 31] required manual input, starting with SMPLify [1] the process has been fully automatized, then improved in [2] to use silhouette annotations,
and eventually extended to multiple views and multiple people [32].    
Regression-based methods, on the other hand, predict 3D pose directly.    
The work of [33] uses sparse linear regression that incorporates a tractable but somewhat weak pose prior. Later approaches use instead deep neural networks, and differ mainly in the nature of their inputs and outputs [9, 6, 10, 11, 12, 13, 14, 7, 15, 17, 16, 34, 35].
Some works start from a pre-detected 2D skeleton [10] while others start from raw images [7].  
Using a 2D skeleton relies on the quality of the underlying 2D keypoint detector and discards appearance details that could help fitting the 3D model to the image. 
Using raw images can potentially make use of this information, but training such models from current 3D indoor datasets might fail to generalize to unconstrained images.    
Hence several papers combine 3D indoor datasets with 2D in-the-wild ones [7, 17, 15, 6, 14, 13, 7, 35, 22]. Methods also differ in their output, with some predicting 3D keypoints directly [10], some predicting the parameters of a 3D human body model [7, 17], and others volumetric heatmaps for the body joints [11].   
Finally, hybrid methods such as SPIN [5] or MTC [17] combine fitting and regression approaches.

一枚の画像から人間の3Dポーズを推定する。  
一枚の画像から3Dポーズを再構築する手法は、3Dポーズの事前情報をどのように組み込むか、また、どのように予測を行うかで異なる。   
フィット法は、SMPL [3]やSCAPE [29]のような3Dボディモデルを仮定し、最適化アルゴリズムを用いて2D観測データにフィットさせる。   
初期のアプローチ[30, 31]では手動入力が必要でしたが，SMPLify[1]からは完全に自動化され，[2]ではシルエットのアノテーションを使用するように改良されました．
そして最終的には，複数のビューと複数の人物にまで拡張されました．   
一方、回帰ベースの手法は、3Dポーズを直接予測する。   
33]の研究では、扱いやすいがやや弱いポーズの事前情報を組み込んだ疎な線形回帰を使用している。その後のアプローチでは、代わりに深層ニューラルネットワークを使用しており、主にその入力と出力の性質が異なる[9, 6, 10, 11, 12, 13, 14, 7, 15, 17, 16, 34, 35]。
事前に検出された2Dスケルトンから始めるものもあれば[10]、生の画像から始めるものもある[7]。 
2Dスケルトンの使用は、基礎となる2Dキーポイント検出器の品質に依存しており、3Dモデルを画像に適合させるのに役立つ外観の詳細は破棄される。  
生の画像を用いることで、この情報を利用できる可能性があるが、現在の3Dインドアデータセットからこのようなモデルをトレーニングしても、制約のない画像に一般化できない可能性がある。   
そのため，いくつかの論文では，3D屋内データセットと2D野生データセットを組み合わせている[7, 17, 15, 6, 14, 13, 7, 35, 22]．また，3Dキーポイントを直接予測する手法もあれば キーポイントを直接予測するもの[10]，3D人体モデルのパラメータを予測するもの[7, 17]，そして体の関節のボリューム・ヒートマップを予測するものなどがある[11]．  
最後に，SPIN [5]やMTC [17]のようなハイブリッド手法は，フィッティングと MTC [17]などのハイブリッド手法は，フィッティングと回帰のアプローチを組み合わせたものです．

3D reconstruction without paired 3D ground-truth.    
While regression methods usually require image datasets paired with 3D ground truth annotations, fitting methods such as SMPLify rely only on 2D annotations by predicting a small number of model parameters and by using a prior learned on independent motion capture data.    
However, their output quality is largely dependent on the initialization, with problematic results for challenging poses (e.g., see Fig. 2 left panel). 
Furthermore, the space of plausible human body poses can be described empirically, by collecting a large number of samples in laboratory conditions [36, 37, 5], but this may lack realism.    
Regression methods [7, 38] can also be learned without requiring images with 3D annotations, by combining 2D datasets with a parametric 3D model and empirical motion capture pose samples, integrating them into their neural network regressor by means of adversarial training.    
However, while the predictions obtained by such methods are plausible, they often do not fit the 2D data very accurately.   
Fitting could be improved by refining this initial solution by means of an optimization based method as in SPIN [8], but empirically we have found that this distorts the pose once again, leading to solutions that are not plausible anymore.

ペアとなる3Dグランドトゥルースがなくても3D再構成が可能です。   
回帰法では、通常、3Dグランドトゥルースのアノテーションとペアになった画像データが必要ですが、SMPLifyのようなフィット法では、少数のモデルパラメータを予測し、独立したモーションキャプチャデータで学習した事前情報を使用することで、2Dアノテーションのみに依存します。   
しかし、これらの手法の出力品質は初期化に大きく依存しており、難易度の高いポーズでは問題のある結果となります（例：図2左パネル参照）。
さらに，もっともらしい人体のポーズの空間は，実験室条件で多数のサンプルを収集することによって経験的に記述することができるが[36, 37, 5]，現実性に欠ける可能性がある．   
回帰法[7, 38]は、パラメトリックな3Dモデルを持つ2Dデータセットと、経験的なモーションキャプチャのポーズサンプルを組み合わせ、それらを敵対的な学習によってニューラルネットワークの回帰器に統合することで、3Dアノテーションを持つ画像を必要とせずに学習することもできる。   
しかし、このような方法で得られた予測はもっともなものですが、2Dデータにはあまり正確にフィットしないことがよくあります。  
この初期解をSPIN[8]のような最適化ベースの手法で改良することで、フィッティングを向上させることができますが、経験的には、この方法ではポーズが再び歪んでしまい、もはやもっともらしくない解になってしまうことがわかっています。

Human pose datasets.   
There are several in-the-wild datasets with sparse 2D pose annotations, including COCO [18], MPII [39], Leeds Sports Pose Dataset (LSP) [40, 41], PennAction [42] and Posetrack [43].    
Furthermore, Dense Pose [44] has introduced a dataset with dense surface point annotations, mapping images to a UV representation of a parametric 3D human model [3].
Compared to annotating 2D keypoints, annotating 3D human poses is much more challenging as there are no easy-to-use or intuitive tools to input the annotations.   
Hence, current 3D annotations are mostly obtained by means of motion capture systems in indoor environments.     
Examples include the Human3.6M dataset [19], Human Eva [45], Panoptic Studio [21], and MPI-INF-3DHP [46].   
These datasets provide 3D motion capture data paired with 2D images, but the images are very controlled.    
3DPW dataset [51] is exceptional by capturing outdoor scenes by a hand-held video camera and IMUs.    
There exists an approach to produce a dataset with 3D pose annotations on Internet photos [2].    
However, the traditional optimization-based fitting method used in this work limits the quality and size of dataset.    
There are also several large scale motion capture datasets that do not have corresponding images at all (e.g. CMU Mocap [47] and KIT [48]).    
These motion capture datasets have recently been reissued in a unified format in the AMASS dataset [49].

人間のポーズのデータセット。  
COCO [18]、MPII [39]、Leeds Sports Pose Dataset (LSP) [40, 41]、PennAction [42]、Posetrack [43]など、スパースな2Dポーズのアノテーションを持つデータセットがいくつか存在する。   
さらに，Dense Pose [44]では，パラメトリックな 3D 人体モデルの UV 表現に画像をマッピングすることで，密な表面点のアノテーションを行ったデータセットを紹介している[3]．   
2次元のキーポイントのアノテーションに比べて，3次元の人物のポーズのアノテーションは，アノテーションを入力するための使いやすい，あるいは直感的なツールがないため，はるかに困難である．   
そのため、現在の3Dアノテーションは、主に屋内環境のモーションキャプチャーシステムを用いて行われている。    
その例として、Human3.6M dataset [19]、Human Eva [45]、Panoptic Studio [21]、MPI-INF-3DHP [46]などがある。
これらのデータセットは、3Dモーションキャプチャーデータと2D画像のペアを提供していますが、画像は非常に制御されています。   
3DPWデータセット[51]は，ハンドヘルドのビデオカメラとIMUを用いて屋外のシーンを撮影した例外的なデータである．   
インターネット上の写真に3Dポーズのアノテーションを付けてデータセットを作成するアプローチがある[2]。   
しかし、この研究で用いられている伝統的な最適化ベースのフィッティング手法では、データセットの品質とサイズが制限される。   
また、大規模なモーションキャプチャのデータセットの中には、対応する画像が全くないものがいくつかある（CMU Mocap[47]やKIT[48]など）。   
これらのモーションキャプチャデータセットは、最近、AMASSデータセットとして統一されたフォーマットで再発行された[49]。





# Result

We consider two applications of EFT: creating pseudo-ground-truth 3D annotations for in-the-wild datasets that natively come only with 2D annotations and post-processing the output of an existing 3D pose regressor to improve it.

ここでは、EFTの2つの応用例を考えてみます。すなわち、2Dアノテーションしかない野生のデータセットに対して擬似的な地表面の3Dアノテーションを作成することと、既存の3Dポーズレグレッサーの出力を後処理して改善することです。

Implementation details.    
For the pose regressor Φ, we use the state-of-the-art SPIN network of [8].    
For EFT, we optimize Eq. (3) using using Adam [53] with the default PyTorch parameters and a small learning rate of 10−6 stopping when the average 2D keypoints re-projection error is less than 2 pixels (usually less than 20 iterations are sufficient).   
We also found beneficial to modify Eq. 3 to ignore the locations of hips and ankles, which are noisy especially for manual annotations, and use instead a term that matches only the 2D orientation of the lower legs (see the appendix for details).  

実装の詳細    
ポーズ・リグレッサーΦには、最先端のSPINネットワーク[8]を使用しています。    
EFTについては，Adam[53]を用いて式(3)を最適化する．(3)をAdam [53]を用いて最適化し、デフォルトのPyTorchパラメータと10-6の小さな学習率で、平均的な2Dキーポイントの再投影エラーが2ピクセル以下になったときに停止します（通常は20回以下の反復で十分です）。  
また、Eq.3を修正して、特にマニュアルアノテーションの場合にノイズとなる腰と足首の位置を無視し、代わりに下肢の2Dオリエンテーションのみにマッチする項を使用することが有益であることがわかりました（詳細は付録を参照）。

Datasets.    
We use the in-the-wild datasets with 2D pose annotations: COCO [18], MPII [39], and LSP [40, 41].   
We consider the default splits as well a the “COCO-Part” subset that [15] uses for training and that contains only instances for which the full set of 12 keypoint annotations are present (occluded instances often miss keypoints).     
To this, we add “COCO-All” containing all samples with at least 5 keypoint annotations.    
We also use datasets with 3D pose annotations, including H36M [19, 54], MPI-INF-3DHP [20], and Panoptic Studio [21].    
Since a multi-view setup is usually required to capture this kind of ground truth, these datasets are collected in laboratory conditions.    
We use the “moshed” version of H36M and MPI-INF-3DHP [7, 8], and produce SMPL fittings for Panoptic Studio DB using the provided 3D keypoints (see the supp. for details).     
The 3DPW dataset [51] is captured outdoor and comes with 3D ground truth obtained by using IMUs and cameras by using IMU sensors and cameras.

データセット    
2Dポーズのアノテーションを持つin-the-wildデータセットを使用します。COCO [18]、MPII [39]、LSP [40, 41]。  
デフォルトの分割に加えて、[15]が学習に使用した "COCO-Part "サブセットも考慮する。このサブセットには、12個のキーポイントアノテーションのフルセットが存在するインスタンスのみが含まれている（隠蔽されたインスタンスはキーポイントを見逃すことが多い）。    
このサブセットに、少なくとも5つのキーポイントのアノテーションがあるすべてのサンプルを含む「COCO-All」を追加しました。   
また，H36M [19, 54]，MPI-INF-3DHP [20]，Panoptic Studio [21]などの3Dポーズアノテーションを持つデータセットも使用している．   
このようなグランドトゥルースを得るためには，通常，マルチビューのセットアップが必要であるため，これらのデータセットは実験室条件で収集されたものである．   
H36MとMPI-INF-3DHP[7, 8]の "moshed "バージョンを使用し，Panoptic Studio DBでは提供された3Dキーポイントを使用してSMPLフィッティングを作成しています（詳細は補足を参照）．    
3DPWデータセット[51]は、屋外で撮影されたもので、IMUセンサーやカメラを用いて得られた3Dグランドトゥルースが付属しています。

# Discussion

We introduced Exemplar Fine-Tuning (EFT), a method to fit a parametric 3D human body model to 2D keypoint annotations.     
Leveraging a trained 3D pose regressor as pose prior conditional on RGB inputs, EFT produces more plausible and accurate fitting outputs than existing methods.   
EFT can be used in post-processing to improve the output of existing 3D pose regressors.    
It can also be used to generate high-quality 3D pseudo-ground-truth annotations for datasets collected in the wild.    
The quality of these labels is sufficient to supervise state-of-the art 3D pose regressors.      
We expect these ‘EFT datasets’ to be of particular interest to the research community because they greatly simplify training 3D pose regressors, avoiding complicated preprocessing or training techniques, as well as the need to mix 2D and 3D annotations.    
We will release the ‘EFT datasets’ to the community, allowing their use in many other tasks, including dense keypoint detection [44], depth estimation [56], or the recognition
of human-object interactions in the wild.

2Dのキーポイントアノテーションにパラメトリックな3D人体モデルをフィットさせる手法であるExemplar Fine-Tuning（EFT）を紹介しました。    
EFTは、訓練された3DポーズレグレッサーをRGB入力に対するポーズ事前条件として活用することで、既存の手法に比べてより妥当で正確なフィット出力を得ることができます。  
EFTは、既存の3Dポーズレグレッサーの出力を向上させるために、後処理で使用することができます。   
また、EFTは、自然界で収集されたデータセットに対して、高品質な3D疑似地表面アノテーションを生成するためにも使用できる。   
これらのラベルの品質は、最先端の3Dポーズレグレッサーを監督するのに十分なものです。     
この「EFTデータセット」は、複雑な前処理や学習技術を必要とせず、2Dと3Dのアノテーションを混在させる必要もないため、3Dポーズレグレッサーの学習を大幅に簡素化することができ、研究者にとっても興味深いものになると期待しています。   
EFTデータセットをコミュニティに公開することで、密なキーポイント検出[44]、深度推定[56]、野生動物における人間と物体のインタラクションの認識など、他の多くのタスクに利用することができます。
人と物の相互作用の認識など、様々なタスクに利用することができます。
