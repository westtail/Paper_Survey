# title
Exemplar Fine-Tuning for 3D Human Model Fitting Towards In-the-Wild 3D Human Pose Estimation  
3D人体モデルフィッティングのための模範解答の微調整 野生の人間の3次元ポーズ推定に向けて

# Abstract
We present Exemplar Fine-Tuning (EFT), a new method to fit a 3D parametric human model to a single RGB input image cropped around a person with 2D keypoint annotations.   
While existing parametric human model fitting approaches, such as SMPLify, rely on the “view-agnostic” human pose priors to enforce the output in a plausible 3D pose space, EFT exploits the pose prior that comes from the specific 2D input observations by leveraging a fully-trained 3D pose regressor.   
We thoroughly compare our EFT with SMPLify, and demonstrate that EFT produces more reliable and accurate 3D human fitting outputs on the same inputs. 
Especially, we use our EFT to augment a large scale in-the-wild 2D keypoint datasets, such as COCO and MPII, with plausible and convincing 3D pose fitting outputs.   
We demonstrate that the pseudo ground-truth 3D pose data by EFT can supervise a strong 3D pose estimator that outperforms the previous state-of-the-art in the standard outdoor benchmark (3DPW), even without using any ground-truth 3D human pose datasets such as Human3.6M.   
Our code and data are available at https://github.com/facebookresearch/eft.

Exemplar Fine-Tuning (EFT)は、2Dのキーポイントアノテーションを持つ人物の周囲を切り取ったRGBの入力画像に、3Dパラメトリック人体モデルをフィットさせる新しい手法です。  
SMPLifyに代表される既存のパラメトリック人体モデル適合手法は、人間の姿勢に関する "ビューアグノスティック "な事前情報に依存して、妥当な3次元姿勢空間での出力を保証しているが、EFTは、完全に訓練された3次元姿勢レグレッサーを利用して、特定の2次元入力観測から得られる姿勢の事前情報を利用する。  
我々のEFTとSMPLifyを徹底的に比較し、EFTが同じ入力に対してより信頼性の高い、正確な3Dフィッティング出力を生成することを実証しました。
特に、我々のEFTを用いて、COCOやMPIIなどの大規模なin-the-wildの2Dキーポイントデータセットを、もっともらしく説得力のある3Dポーズフィッティング出力で補強しました。  
その結果、EFTによる擬似的な3Dポーズデータは、屋外での標準的な撮影において、従来の最先端技術を凌駕する強力な3Dポーズ推定器を監督することができることを示しました。その結果、Human3.6Mのような人間の3Dポーズデータを使用しなくても、標準的な屋外ベンチマーク（3DPW）において、従来の最先端技術を凌駕する強力な3Dポーズ推定量が得られることを実証しました。  
我々のコードとデータは、https://github.com/facebookresearch/eft。

## Introduction
We consider the problem of reconstructing the pose of humans in 3D from single 2D images, a key task in applications such as human action recognition, human-machine interaction and virtual and augmented reality.   
Since individual 2D images do not contain sufficient information for 3D reconstruction, algorithms must supplement the missing information by a learned prior on the plausible 3D poses of the human body.   
Established approaches such as SMPLify [1, 2] cast this as fitting the parameters of a 3D human model [3, 4, 5] to the location of 2D keypoints, manually or automatically annotated in images. 
They regularize the solution by means of a “view-agnostic” 3D pose prior, which incurs some limitations: fitting ignores the RGB images themselves, the 3D priors, often learned separately in laboratory conditions, lack realism, and balancing between the prior and the data term (e.g., 2D keypoint error) is difficult.

この問題は、人間の行動認識、ヒューマンマシンインタラクション、バーチャルリアリティや拡張現実などのアプリケーションにおいて重要な課題です。  
個々の2D画像には、3D再構成に必要な十分な情報が含まれていないため、アルゴリズムは、人体の妥当な3Dポーズに関する学習済みの事前情報によって、不足している情報を補う必要がある。  
SMPLify [1, 2]のような確立されたアプローチでは，3D人体モデル [3, 4, 5]のパラメータを2Dキーポイントの位置にフィットさせることで，この作業を行っている．SMPLify [1, 2]などの確立されたアプローチは，3D人体モデル [3, 4, 5]のパラメータを，画像に手動または自動で注釈された2Dキーポイントの位置に適合させるものである．  
彼らは，"視界にとらわれない "3Dポーズ事前情報によって解を正則化している．これにはいくつかの制限があります．例えば，フィッティングでは，RGB画像そのものや3Dプリオールを無視します．3Dプリオールは、実験室環境で個別に学習されることが多いため、リアリティに欠け、また、プリオールとデータ項の間のバランスが取れていない。データ項（2Dキーポイントエラーなど）とのバランスをとるのが難しい。

##  Discussion
We introduced Exemplar Fine-Tuning (EFT), a method to fit a parametric 3D human body model to 2D keypoint annotations.   
Leveraging a trained 3D pose regressor as pose prior conditional on RGB inputs, EFT produces more plausible and accurate fitting outputs than existing methods.   
EFT can be used in post-processing to improve the output of existing 3D pose regressors.   
It can also be used to generate high-quality 3D pseudo-ground-truth annotations for datasets collected in the wild.   
The quality of these labels is sufficient to supervise state-of-the art 3D pose regressors.   
We expect these ‘EFT datasets’ to be of particular interest to the research community because they greatly simplify training 3D pose regressors, avoiding complicated preprocessing or training techniques, as well as the need to mix 2D and 3D annotations.   
We will release the ‘EFT datasets’ to the community, allowing their use in many other tasks, including dense keypoint detection [44], depth estimation [56], or the recognition of human-object interactions in the wild.

パラメトリックな3D人体モデルを2Dキーポイントアノテーションに適合させる手法であるExemplar Fine-Tuning（EFT）を紹介しました。  
EFTは、訓練された3DポーズレグレッサーをRGB入力に対するポーズ事前条件として活用することで、既存の手法に比べてより妥当で正確なフィット出力を得ることができます。  
EFTは、既存の3Dポーズレグレッサーの出力を向上させるために、後処理で使用することができます。  
また、EFTは、自然界で収集されたデータセットに対して、高品質な3D疑似地表面アノテーションを生成するためにも使用できる。  
これらのラベルの品質は、最先端の3Dポーズレグレッサーを監督するのに十分なものである。  
この「EFTデータセット」は、複雑な前処理や学習技術を必要とせず、2Dと3Dのアノテーションを混在させる必要もないため、3Dポーズレグレッサーの学習を大幅に簡素化することができ、研究者にとっても興味深いものになると期待しています。  
EFTデータセットをコミュニティに公開することで、密なキーポイント検出[44]、深度推定[56]、野生での人間と物体の相互作用の認識など、他の多くのタスクに利用できるようになる。
