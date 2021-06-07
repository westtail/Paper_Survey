# title 
Exemplar Fine-Tuning for 3D Human Model Fitting Towards In-the-Wild 3D Human Pose Estimation  
3次元人体モデルのフィッティングのための模範的な微調整と、実世界での3次元人体ポーズの推定

## Abstract
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

## Introduction
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

