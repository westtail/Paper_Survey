# title
Deep Learning-Based Human Pose Estimation A Survey  
深層学習を用いた人物ポーズ推定に関する調査

## Abstract
Human pose estimation aims to locate the human body parts and build human body representation (e.g., body skeleton) from input data such as images and videos.   
It has drawn increasing attention during the past decade and has been utilized in a wide range of applications including human-computer interaction, motion analysis, augmented reality, and virtual reality.     
Although the recently developed deep learning-based solutions have achieved high performance in human pose estimation, there still remain challenges due to insufficient training data, depth ambiguities, and occlusion.   
The goal of this survey paper is to provide a comprehensive review of recent deep learning-based solutions for both 2D and 3D pose estimation via a systematic analysis and comparison of these solutions based on their input data and inference procedures.   
More than 240 research papers since 2014 are covered in this survey.   
Furthermore, 2D and 3D human pose estimation datasets and evaluation metrics are included.   
Quantitative performance comparisons of the reviewed methods on popular datasets are summarized and discussed.   
Finally, the challenges involved, applications, and future research directions are concluded.    
We also provide a regularly updated project page: https://github.com/zczcwh/DL-HPE

人体姿勢推定は、画像や動画などの入力データから、人体の部位を特定し、人体表現（骨格など）を構築することを目的としています。  
これは、過去10年間に注目を集め、人間とコンピュータのインタラクション、動作解析、拡張現実感、仮想現実感など、幅広い用途に利用されています。    
最近開発された深層学習を用いたソリューションは，人間の姿勢推定において高い性能を達成しているが，学習データの不足，深度の曖昧さ，オクルージョンなどの課題が残っている．  
この調査論文の目的は、入力データと推論手順に基づいたこれらのソリューションの体系的な分析と比較を通じて、2Dと3Dの両方のポーズ推定のための最近の深層学習ベースのソリューションの包括的なレビューを提供することです。  
本調査では、2014年以降の240以上の研究論文を対象としています。  
さらに、2Dおよび3Dの人間のポーズ推定データセットと評価指標も含まれています。  
一般的なデータセットを用いて、検討された手法の定量的な性能比較を行っています。定量的な性能比較を行い、議論しています。  
最後に、課題、アプリケーション、および将来の研究の方向性について 結論を述べています。   
また、定期的にプロジェクトページを更新しています。https://github.com/zczcwh/DL-HPE

Index Terms—Survey of human pose estimation, 2D and 3D pose estimation, deep learning-based pose estimation, pose estimation datasets, pose estimation metrics

索引用語-人間のポーズ推定に関する調査、2Dおよび3Dポーズ推定、深層学習に基づくポーズ推定、ポーズ推定データセット、ポーズ推定メトリクス

## 1 INTRODUCTION
HUMAN pose estimation (HPE), which has been extensively studied in computer vision literature, involves estimating the configuration of human body parts from input data captured by sensors, in particular images and videos.   
HPE provides geometric and motion information of the human body which has been applied to a wide range of applications (e.g., human-computer interaction, motion analysis, augmented reality (AR), virtual reality (VR), healthcare, etc.).    
With the rapid development of deep learning solutions in recent years, such solutions have been shown to outperform classical computer vision methods in various tasks including image classification [1], semantic segmentation [2], and object detection [3].    
Significant progress and remarkable performance have already been made by employing deep learning techniques in HPE tasks.    
However, challenges such as occlusion, insufficient training data, and depth ambiguity still pose difficulties to be overcome.    
2D HPE from images and videos with 2D pose annotations is easily achievable and high performance has been reached for the human pose estimation of a single person using deep learning techniques.    
More recently, attention has been paid to highly occluded multi-person HPE in complex scenes.    
In contrast, for 3D HPE, obtaining accurate 3D pose annotations is much more difficult than its 2D counterpart.    
Motion capture systems can collect 3D pose annotation in controlled lab environments; however, they have limitations for in-thewild environments.     
For 3D HPE from monocular RGB images and videos, the main challenge is depth ambiguities.    
In multi-view settings, viewpoints association is the key issue that needs to be addressed.   
Some works have utilized sensors such as depth sensor, inertial measurement units (IMUs), and radio frequency devices, but these approaches are usually not cost-effective and require special purpose hardware.    
Given the rapid progress in HPE research, this article attempts to track recent advances and summarize their achievements in order to provide a clear picture of current research on deep learning-based 2D and 3D HPE.

人間の姿勢推定（HPE）は，画像や動画などのセンサで取得した入力データから，人間の体の部位の形状を推定するもので，コンピュータビジョンの分野で広く研究されている．  
HPEは、人体の幾何学的情報と動きの情報を提供し、幅広い用途（ヒューマン・コンピュータ・インタラクション、動作解析、拡張現実（AR）、仮想現実（VR）、ヘルスケアなど）に応用されています。   
近年、深層学習ソリューションが急速に発展しており、そのようなソリューションは、画像分類[1]、セマンティックセグメンテーション[2]、物体検出[3]などの様々なタスクにおいて、古典的なコンピュータビジョン手法を凌駕することが示されています。   
HPEタスクに深層学習技術を採用することで、すでに大きな進歩と顕著な性能が得られています。   
しかし、オクルージョン、不十分な学習データ、深度の曖昧さなどの課題は、克服すべき困難をもたらしています。   
しかし、2次元の姿勢情報が付与された画像や動画からの2次元HPEは容易に実現可能であり、深層学習技術を用いた1人の人物の姿勢推定では高い性能が得られています。   
最近では、複雑なシーンでのオクルージョンの高い複数人のHPEにも注目が集まっています。   
一方、3D HPEでは、正確な3Dポーズアノテーションを得ることは、2D HPEに比べて非常に困難です。   
モーションキャプチャシステムは、制御された実験室環境では3Dポーズアノテーションを収集することができますが、野生の環境では限界があります。    
単眼のRGB画像やビデオから3D HPEを行う場合、主な課題は奥行きの曖昧さです。   
また、多視点環境では、視点の関連付けが重要な課題となります。  
これまでに、深度センサー、慣性計測ユニット（IMU）、高周波デバイスなどのセンサーを利用する方法がありましたが、これらのアプローチは通常、費用対効果が低く、専用のハードウェアを必要とします。   
HPEの研究が急速に進展していることを踏まえ、本稿では、深層学習ベースの2Dおよび3D HPEに関する現在の研究の全体像を把握するために、最近の進歩を追跡し、その成果をまとめようとしている。

## 1.1 Previous surveys and our contributions
* 過去の調査と私たちの貢献

Table 1 lists the related surveys and reviews previously reported on HPE.    
Among them, [4] [5] [6] [7] focus on the general field of visual-based human motion capture methods and their implementations including pose estimation, tracking, and action recognition.    
Therefore, pose estimation is only one of the topics covered in these surveys.    
The research works on 3D human pose estimation before 2012 are reviewed in [8].    
The body parts parsing-based methods for single-view and multi-view HPE are reported in [9].     
These surveys published during 2001-2015 mainly focus on conventional methods without deep learning.    
A survey on both traditional and deep learning-based methods related to HPE is presented in [10].   
However, only a handful of deep learning-based approaches are included.    
The survey in [11] covers 3D HPE methods with RGB inputs.    
The survey in [13] only reviews 2D HPE methods and analyzes model interpretation.     
Monocular HPE from the classical to recent deep learning-based methods (till 2019) is summarized in [12].     
However, it only covers 2D HPE and 3D single-view HPE from monocular images/videos.    
Also, no extensive performance comparison is given.    
This survey aims to address the shortcomings of the previous surveys in terms of providing a systematic review of the recent deep learning-based solutions to 2D and 3D HPE but also covering other aspects of HPE including the performance evaluation of (2D and 3D) HPE methods on popular datasets, their applications, and comprehensive discussion.     
The key points that distinguish this survey from the previous ones are as follows:   

* A comprehensive review of recent deep learning-based 2D and 3D HPE methods (up to 2020) is provided by categorizing them according to 2D or 3D scenario, singleview or multi-view, from monocular images/videos or other sources, and learning paradigm.   
* Extensive performance evaluation of 2D and 3D HPE methods. We summarize and compare reported performances of promising methods on common datasets based on their categories.    
The comparison of results provides cues for the strength and weakness of different methods, revealing the research trends and future directions of HPE.
* An overview of a wide range of HPE applications, such as gaming, surveillance, AR/VR, and healthcare.
* An insightful discussion of 2D and 3D HPE is presented in terms of key challenges in HPE pointing to potential future research towards improving performance.

These contributions make our survey more comprehensive, up-to-date, and in-depth than previous survey papers.  

表1は、HPEに関して以前に報告された関連する調査とレビューの一覧です。   
その中でも、[4] [5] [6] [7]は、ポーズ推定、トラッキング、アクション認識など、ビジュアルベースの人間のモーションキャプチャ方法とその実装の一般的な分野に焦点を当てています。   
したがって，ポーズ推定は，これらの調査で取り上げられているトピックの1つに過ぎない．   
2012 年以前に行われた人間の 3D ポーズ推定に関する研究を [8]でレビューしている。   
また、シングルビューおよびマルチビューHPEのためのボディパーツパーシングベースの手法は[9]に報告されている。    
2001年から2015年に発表されたこれらの調査は、主に深層学習を伴わない従来の手法に焦点を当てている。   
HPEに関連する従来の手法と深層学習ベースの手法の両方に関する調査は[10]に示されている。  
しかし、深層学習ベースのアプローチはほんの一握りしか含まれていません。   
11]の調査では、RGB入力の3D HPE手法を対象としています。   
13]の調査では、2D HPE法をレビューし、モデルの解釈を分析するのみである。    
古典的な手法から最近の深層学習ベースの手法までの単眼HPE（2019年まで）は[12]にまとめられている。    
しかし、単眼画像/動画からの2D HPEと3D単眼HPEのみを対象としている。   
また、広範な性能比較も行われていない。   
本調査は、2Dおよび3D HPEに対する最近の深層学習ベースのソリューションの体系的なレビューを提供するという点で、これまでの調査の欠点を解決することを目的としているが、一般的なデータセットでの（2Dおよび3D）HPE手法の性能評価、その応用、および包括的な議論を含むHPEの他の側面もカバーしている。    
本調査がこれまでの調査と異なるポイントは、以下の通りです。 

* 最近の深層学習に基づく2Dおよび3D HPE手法（2020年まで）を、2Dまたは3Dのシナリオ、単眼または多眼、単眼画像/動画または他のソースから、および学習パラダイムに応じて分類し、包括的にレビューしています。  
* 2Dおよび3D HPE手法の広範な性能評価。一般的なデータセットにおいて、有望な手法の性能をカテゴリー別にまとめ、比較しています。  
結果を比較することで、異なる手法の強みと弱みを知る手がかりとなり、HPEの研究動向と今後の方向性が明らかになります。
* ゲーム、監視、AR/VR、ヘルスケアなど、幅広いHPEアプリケーションの概要を紹介。
* 2Dと3DのHPEについて、HPEにおける重要な課題を提示し、性能向上に向けた将来の研究の可能性を示唆しています。   

これらの貢献により、本調査はこれまでの調査論文よりも包括的で、最新かつ詳細なものとなっています。

## 1.2 Organization
編成について

In the following sections, we will cover various aspects of recent advances in HPE with deep learning.     
We first overview the human body modeling techniques in § 2.    
Then, HPE is divided into two main categories: 2D HPE (§ 3) and 3D HPE (§ 4).   
Fig. 1 shows the taxonomy of deep learning methods for HPE.    
According to the number of people, 2D HPE methods are categorized into single-person and multi-person settings.    
For single-person methods (§ 3.1), there are two categories of deep learning-based methods:   
(1) regression methods, which directly build a mapping from input images to body joint coordinates by employing deep learning-based regressors;    
(2) body part detection methods, which consist of two steps: the first step involves generating heatmaps of keypoints (i.e., joints) for body part localization, and the second step involves assembling these detected keypoints into whole body pose or skeleton.    
For multi-person methods (§ 3.2), there are also two types of deep learning-based methods:   
(1) top-down methods, which construct human body poses by detecting the people first and then utilizing single-person HPE to predict the keypoints for each person;   
(2) bottom-up methods, which first detect body keypoints without knowing the number of people, then group the keypoints into individual poses.     
3D HPE methods are classified according to the input source types: monocular RGB images and videos (§ 4.1), or other sensors (e.g., inertial measurement unit sensors, § 4.2).     
The majority of these methods use monocular RGB images and videos, and they are further divided into single-view and multi-view methods.     
Single-view methods are then separated by single-person versus multi-person.     
Multi-view settings are deployed mainly for multi-person pose estimation. Hence, single-person or multi-person is not specified in this category.    
Next, depending on the 2D and 3D HPE pipelines, the datasets and evaluation metrics commonly used are summarized followed by a comparison of results of the promising methods (§ 5).      
In addition, various applications of HPE such as AR/VR are mentioned (§ 6).    
Finally, the paper ends by an insightful discussion of some promising directions for future research (§ 7).

以下のセクションでは、深層学習を用いたHPEの最近の進歩について様々な角度から取り上げます。    
まず、§2で人体のモデリング技術を概観する。   
そして、HPEを2D HPE（§3）と3D HPE（§4）の2つに大別する。  
図1は、HPEのための深層学習手法の分類法を示している。   
2D HPEの手法は、人数に応じて、一人用と複数人用に分類される。   
一人用の手法（§3.1）では、深層学習を用いた手法は2つに分類される。  
(1)回帰法：深層学習を用いた回帰器を用いて、入力画像から体の関節座標へのマッピングを直接構築する方法。   
(2) 体の部位を検出する手法で、体の部位を特定するためのキーポイント（関節）のヒートマップを生成するステップと、そのヒートマップを組み合わせて体の部位を検出するステップの2つがある。第2段階では、検出されたキーポイントを全身のポーズやスケルトンに組み立てる。   
また、多人数向けの手法（§3.2）では、深層学習を用いた2種類の手法があります。  
(1)トップダウン方式は、まず人物を検出し、一人用のHPEを利用して各人物のキーポイントを予測することで人体のポーズを構築する。  
(2)人数がわからない状態で体のキーポイントを検出し、そのキーポイントを個々のポーズにグループ化するボトムアップ方式。    
3D HPE手法は、入力ソースの種類によって、単眼のRGB画像や動画（§4.1）、またはその他のセンサ（例えば、慣性計測ユニットセンサ、§4.2）に分類される。    
これらの手法の多くは，単眼のRGB画像や動画を用いるものであり，さらに，単眼の手法と多眼の手法に分けられる．    
シングルビュー方式は、その後、一人用と複数人用で分けられる。    
マルチビュー方式は、主に多人数のポーズ推定に用いられる。したがって、このカテゴリーでは、single-personかmulti-personかは特定されません。   
次に、2Dと3DのHPEパイプラインに応じて、一般的に使用されているデータセットと評価指標をまとめ、有望な手法の結果を比較する（§5）。     
さらに、AR/VRなどのHPEの様々なアプリケーションについても言及する（§6）。   
最後に，今後の研究の方向性について，洞察に満ちた議論をして本稿を終える（§7）。

## 2 HUMAN BODY MODELING
* 人体のモデリング  
Human body modeling is an important aspect of HPE in order to represent keypoints and features extracted from input data.    
For example, most HPE methods use an N-joints rigid kinematic model.    
A human body is a sophisticated entity with joints and limbs, and contains body kinematic structure and body shape information.    
In typical methods, a model-based approach is employed to describe and infer human body pose, and render 2D and 3D poses.     
There are typically three types of models for human body modeling, i.e., kinematic model (used for 2D/3D HPE), planar model (used for 2D HPE) and volumetric model (used for 3D HPE), as shown in Fig. 2.     
In the following sections, a description of these models is provided covering different representations.  

人体のモデリングは、入力データから抽出したキーポイントや特徴を表現するために、HPEの重要な要素です。   
例えば、多くのHPE手法では、N関節の剛体運動モデルを使用しています。   
人体は、関節や手足を持つ精巧な実体であり、体の運動構造や体の形状情報を含んでいます。   
一般的な手法では、人体の姿勢を記述・推定し、2Dおよび3Dポーズをレンダリングするために、モデルベースのアプローチが採用されています。    
人体モデルには、図2に示すように、キネマティックモデル（2D/3D HPEに使用）、プラナーモデル（2D HPEに使用）、ボリュームモデル（3D HPEに使用）の3種類のモデルがある。    
以下では、これらのモデルについて、異なる表現方法を説明します。

##  2.1 Kinematic model   
* キネマティックモデル   
The kinematic model, also called skeleton-based model [12] or kinematic chain model [14], as shown in Fig. 2 (a), includes a set of joint positions and the limb orientations to represent the human body structure.     
The pictorial structure model (PSM) [15] is a widely used graph model, which is also known as the tree-structured model.     
This flexible and intuitive human body model is successfully utilized in 2D HPE [16] [17] and 3D HPE [18] [19].     
Although the kinematic model has the advantage of flexible graph-representation, it is limited in representing texture and shape information.

キネマティック・モデルは，図2(a)に示すように，スケルトン・ベース・モデル[12]やキネマティック・チェーン・モデル[14]とも呼ばれ，人体の構造を表現するための関節位置と四肢の向きのセットを含む．    
PSM（Pictorial Structure Model）[15]は，広く使われているグラフモデルで，木構造モデルとも呼ばれています．    
この柔軟で直感的な人体モデルは，2D HPE [16] [17]や3D HPE [18] [19]でうまく活用されている．    
キネマティックモデルは、柔軟なグラフ表現という利点があるものの、テクスチャや形状の情報を表現するには限界がある。

## 2.2 Planar model 
* 平面モデル  
Other than the kinematic model to capture the relations between different body parts, the planar model is used to represent the shape and appearance of a human body as shown in Fig. 2 (b).   
In the planar model, body parts are usually represented by rectangles approximating the human body contours.    
One example is the cardboard model [20], which is composed of body part rectangular shapes representing the limbs of a person.     
One of the early works [21] used the cardboard model in HPE.    
Another example is Active Shape Model (ASM) [22], which is widely used to capture the full human body graph and the silhouette deformations using principal component analysis [23] [24].   

人体の各部位の関係を表現するには、運動学的モデルのほかに、図2（b）に示すように、平面モデルが用いられる。  
平面モデルでは，体の各部位は通常，人体の輪郭に近似した長方形で表現される．   
例えば，ダンボールモデル[20]は，人の手足を表すボディパーツの長方形で構成されている．    
初期の作品[21]では，このダンボールモデルをHPEで使用していた．   
もう一つの例は、アクティブ・シェイプ・モデル（ASM）[22]であり、主成分分析を用いて人体の全グラフとシルエットの変形を捉えるために広く使用されている[23][24]。

## 2.3 Volumetric models
* ボリューメトリックモデル  
With the increasing interest in 3D human reconstruction, many human body models have been proposed for a wide variety of human body shapes.     
We briefly discuss several popular 3D human body models used in deep learningbased 3D HPE methods for recovering 3D human mesh.   
The volumetric model representation is depicted in Fig. 2 (c).   

人体の3次元復元への関心の高まりに伴い、様々な人体形状に対応した多くの人体モデルが提案されている。    
ここでは、深層学習ベースの3D HPE法で3D人体メッシュを復元する際に使用される、いくつかの一般的な3D人体モデルについて簡単に説明します。  
図2(c)にボリュームモデルの表現を描いています。  

### SMPL: Skinned Multi-Person Linear model
Skinned Multi-Person Linear model [25] is a skinned vertex-based model which represents a broad range of human body shapes.    
SMPL can be modeled with natural pose-dependent deformations exhibiting soft-tissue dynamics.     
To learn how people deform with pose, there are 1786 high-resolution 3D scans of different subjects of poses with template mesh in SMPL to optimize the blend weights [26], pose-dependent blend shapes, the mean template shape, and the regressor from vertices to joint locations.      
SMPL is easy to deploy and compatible with existing rendering engines, therefore is widely adopted in 3D HPE methods.

Skinned Multi-Person Linear model [25]は，様々な人体形状を表現することができる，頂点ベースのスキンシップモデルです．   
SMPLは，軟部組織のダイナミクスを示す自然なポーズ依存の変形をモデル化することができます．    
人がどのように姿勢に応じて変形するかを知るために，SMPLのテンプレートメッシュを用いた様々な被験者の1786件の高解像度3Dスキャンデータを用いて，ブレンドウェイト[26]，姿勢依存のブレンド形状，平均テンプレート形状，頂点から関節位置への回帰子を最適化した．     
SMPLは、導入が容易で、既存のレンダリングエンジンとの互換性があるため、3D HPE手法に広く採用されています。

### DYNA: Dynamic Human Shape in Motion
Dynamic Human Shape in Motion [27] model attempts to represent realistic soft-tissue motions for various body shapes.     
Motion related soft-tissue deformation is approximated by a low-dimensional linear subspace.     
In order to predict the low-dimensional linear coefficients of softtissue motion, the velocity and acceleration of the whole body, the angular velocities and accelerations of the body parts, and the soft-tissue shape coefficients are used.     
Moreover, DYNA leverages body mass index (BMI) to produce different deformations for people with different shapes.

Dynamic Human Shape in Motion [27] モデルは，様々な体形に対する現実的な軟組織の動きを表現しようとするものである．    
運動に関連する軟組織の変形は，低次元の線形部分空間で近似されます．    
軟部組織の動きの低次元の線形係数を予測するために、全身の速度と加速度、身体部位の角速度と加速度、および軟部組織の形状係数が使用されます。    
さらに、DYNAはBMI（Body Mass Index）を活用して、形状の異なる人に対して異なる変形を生成する。

### Stitched Puppet Model 
Stitched Puppet Model [28] is a part-based graphical model integrated with a realistic body model.    
Different 3D body shapes and pose-dependent shape variations can be translated to the corresponding graph nodes representation.     
Each body part is represented by its own low-dimensional state space.     
The body parts are connected via pairwise potentials between nodes in the graph that ”stitch” the parts together. In general, part connection via potential functions is performed by using message passing algorithms such as Belief Propagation (BP).    
To solve the problem that the state space of each part cannot be easily discretized to apply discrete BP, a max-product BP via a particle-based D-PMP model [29] is applied.

Stitched Puppet Model [28]は，パーツベースのグラフィックモデルとリアルなボディモデルを統合したものである．   
さまざまな3Dボディ形状や、ポーズに依存した形状変化を、対応するグラフノード表現に変換することができます。    
各ボディ・パーツは，それぞれ低次元の状態空間で表現されます．    
体のパーツは、グラフ内のノード間のペアの電位を介して接続され、パーツ同士を「つなぎ合わせる」ことができます。一般に、ポテンシャル関数によるパーツの接続は、BP（Belief Propagation）などのメッセージパッシングアルゴリズムを用いて行われます。   
ここでは，離散BPを適用するために各パーツの状態空間を容易に離散化できないという問題を解決するために，粒子ベースのD-PMPモデル[29]による最大積BPを適用している．

### Frankenstein & Adam: 
The Frankenstein model [30] produces human motion parameters not only for body motion but also for facial expressions and hand gestures.    
This model is generated by blending models of the individual component meshes: SMPL [25] for the body, FaceWarehouse [31] for the face, and an artist rigged for the hand.   
All transform bones are merged into a single skeletal hierarchy while the native parameterization of each component is kept to express identity and motion variations.   
The Adam model [30] is optimized by the Frankenstein model using a largescale capture of people’s clothes.     
With the ability to express human hair and clothing geometry, Adam is more suitable to represent human under real-world conditions.

Frankensteinモデル[30]は，体の動きだけでなく，顔の表情や手のジェスチャーにも対応した人間のモーションパラメータを生成します．   
このモデルは、個々のコンポーネントメッシュのモデルをブレンドして生成されます。ボディにはSMPL [25]，顔にはFaceWarehouse [31]，手にはアーティストが作成したリグが使用されています．  
すべてのトランスフォームボーンは、単一の骨格階層にマージされ、各コンポーネントのネイティブなパラメータ化は、アイデンティティとモーションバリエーションを表現するために維持されます。  
アダムモデル[30]は、フランケンシュタインモデルによって、人々の衣服の大規模なキャプチャを使用して最適化されています。    
人間の髪の毛や服の形状を表現できるAdamは、実世界の状況下で人間を表現するのに適しています。

### GHUM & GHUML(ite): 
A fully trainable end-to-end deep learning pipeline is proposed in [32] to model statistical and articulated 3D human body shape and pose.     
GHUM is the moderate resolution version and GHUML is the low resolution version.    
GHUM and GHUML are trained by high resolution full-body scans (over 60,000 diverse human configurations in their dataset) in a deep variational auto-encoder framework.      
They are able to infer a host of components such as non-linear shape spaces, pose-space deformation correctives, skeleton joint center estimators, and blend skinning function [26].

GHUMとGHUML(ite)は、統計的かつ関節的な3D人体形状とポーズをモデル化するために、完全に学習可能なエンドツーエンドの深層学習パイプラインを[32]で提案しています。    
GHUMは中程度の解像度のバージョン、GHUMLは低解像度のバージョンです。   
GHUMとGHUMLは、高解像度のフルボディスキャン（彼らのデータセットでは60,000以上の多様な人間の構成）を用いて、深層変分オートエンコーダーのフレームワークで学習されます。     
GHUMとGHUMLは，高解像度のフルボディスキャン（同社のデータセットでは6万人以上の多様な人間の構成）を用いて，深層変分オートエンコーダーの枠組みで学習されており，非線形形状空間，ポーズ空間の変形補正，スケルトンの関節中心の推定，ブレンドスキニング機能など，多数のコンポーネントを推論することができる[26]

## 3 2D HUMAN POSE ESTIMATION 2次元人体姿勢推定
2D HPE methods estimate the 2D position or spatial location of human body keypoints from images or videos.     
Traditional 2D HPE methods adopt different hand-crafted feature extraction techniques [33] [34] for body parts, and these early works describe human body as a stick figure to obtain global pose structures.    
Recently, deep learning-based approaches have achieved a major breakthrough in HPE by improving the performance significantly. 
In the following, we review deep learning-based 2D HPE methods with respect to singleperson and multi-person scenarios.    

2D HPE法は、画像や動画から人体のキーポイントの2D位置または空間的な位置を推定する。    
従来の2D HPE手法では、体のパーツに対して異なる手作りの特徴抽出技術[33][34]を採用しており、これらの初期の作品では、グローバルなポーズ構造を得るために人体を棒人間として記述している。   
近年、深層学習に基づくアプローチは、性能を大幅に向上させることで、HPEにおいて大きなブレークスルーを達成している。
以下では、深層学習ベースの2D HPE手法を、1人用と複数人用のシナリオに関してレビューする。

### 3.1 2D single-person pose estimation 2D一人称のポーズ推定
2D single-person pose estimation is used to localize human body joint positions when the input is a single-person image.    
If there are more than one person, the input image is cropped first so that there is only one person in each cropped patch (or sub-image).     
This process can be achieved automatically by an upper-body detector [35] or a full-body detector [3].     
In general, there are two categories for single-person pipelines that employ deep learning techniques: regression methods and body part detection methods.    
Regression methods apply an end-to-end framework to learn a mapping from the input image to body joints or parameters of human body models [36].      
The goal of body part detection methods is to predict approximate locations of body parts and joints [37] [38], which are normally supervised by heatmaps representation [39] [40].      
Heatmap-based frameworks are now widely used in 2D HPE tasks.      
The general frameworks of 2D single-person HPE methods are depicted in Fig. 3.

2D一人称ポーズ推定は、入力が一人の人物の画像である場合に、人体の関節位置を特定するために使用されます。   
複数の人物がいる場合は、まず入力画像が切り取られ、切り取られた各パッチ（またはサブ画像）に1人の人物だけが写るようにする。    
この処理は，上半身の検出器[35]や全身の検出器[3]によって自動的に行うことができる．    
一般的に、深層学習技術を採用した1人用パイプラインには、回帰法と身体部位検出法の2つのカテゴリーがあります。   
回帰法は、エンド・ツー・エンドのフレームワークを適用して、入力画像から体の関節や人体モデルのパラメータへのマッピングを学習するものである[36]。     
身体部位検出法の目的は、身体部位や関節のおおよその位置を予測することであり[37][38]、通常はヒートマップ表現によって監督される[39][40]。     
ヒートマップベースのフレームワークは、現在、2D HPEタスクで広く使用されている。     
2D一人用HPE手法の一般的なフレームワークは、図3に示されている。


### 3.1.1 Regression methods 回帰法
There are many works based on the regression framework (e.g., [36] [41] [42] [43] [44] [45] [46] [47] [48] [49]) to predict joint coordinates from images as shown in Fig. 3    
(a). Using AlexNet [1] as the backbone, Toshev and Szegedy [36] proposed a cascaded deep neural network regressor named DeepPose to learn keypoints from images.     
Due to the impressive performance of DeepPose, the research paradigm of HPE began to shift from classic approaches to deep learning, in particular convolutional neural networks (CNNs).      
Based on GoogLeNet [50], Carreira et al. [42] proposed an Iterative Error Feedback (IEF) network, which is a self-correcting model to progressively change an initial solution by injecting the prediction error back to the input space.     
Sun et al. [43] introduced a structure-aware regression method called ”compositional pose regression” based on ResNet-50 [51].    
This method adopts a re-parameterized and bone-based representation that contains human body information and pose structure, instead of the traditional joint-based representation.      
Luvizon et al. [44] proposed an end-to-end regression approach for HPE using soft-argmax function to convert feature maps into joint coordinates in a fully differentiable framework.     
A good feature that encodes rich pose information is critical for regression-based methods.      
One popular strategy to learn better feature representation is multi-task learning [52].    
By sharing representations between related tasks (e.g., pose estimation and pose-based action recognition), the model can generalize better on the original task (pose estimation).     
Following this direction, Li et al. [46] proposed a heterogeneous multi-task framework that consists of two tasks: predicting joints coordinates from full images by building a regressor and detecting body parts from image patches using a sliding window.     
Fan et al. [47] proposed a Dual-Source (i.e., image patches and full images) Deep Convolutional Neural Network (DS-CNN) for two tasks: joint detection which determines whether a patch contains a body joint, and joint localization which finds the exact location of the joint in the patch. 
Each task corresponds to a loss function, and the combination of two tasks leads to improved results.     
Luvizon et al. [48] learned a multi-task network to jointly handle 2D/3D pose estimation and action recognition from video sequences. 

回帰法のフレームワークを用いて，図3に示すように，画像から関節の座標を予測する研究は数多く存在する（例えば，[36] [41] [42] [43] [44] [45] [46] [47] [48] [49]）．   
(a). Toshev and Szegedy [36]は，AlexNet [1]をバックボーンとして，DeepPoseという名前のカスケード接続された深層ニューラルネットワークを用いて，画像からキーポイントを学習することを提案した．    
DeepPoseの優れた性能により、HPEの研究パラダイムは、古典的なアプローチから深層学習、特に畳み込みニューラルネットワーク（CNN）へと移行し始めました。     
Carreiraら[42]は、GoogLeNet[50]をベースに、予測誤差を入力空間に戻すことで初期解を徐々に変化させる自己修正モデルであるIterative Error Feedback（IEF）ネットワークを提案しました。    
Sunら[43]は，ResNet-50[51]をベースにした "compositional pose regression "と呼ばれる構造を考慮した回帰手法を導入した．   
この手法では、従来の関節ベースの表現の代わりに、人体情報とポーズ構造を含む再パラメータ化された骨ベースの表現を採用している。     
Luvizonら[44]は、完全に微分可能なフレームワークで、特徴マップを関節座標に変換するために、soft-argmax関数を用いたHPEのエンド・ツー・エンド回帰アプローチを提案した。    
豊富なポーズ情報を符号化する優れた特徴は、回帰ベースの手法にとって重要である。     
より良い特徴表現を学習するための一般的な戦略の1つは、マルチタスク学習である[52]。   
関連するタスク（例えば、ポーズ推定とポーズベースの行動認識）の間で表現を共有することで、モデルは元のタスク（ポーズ推定）でよりよく一般化することができる。    
この方向性に沿って、Liら[46]は、リグレッサを構築してフル画像から関節座標を予測するタスクと、スライディングウィンドウを用いて画像パッチからボディパーツを検出するタスクの2つから構成されるヘテロジニアス・マルチタスク・フレームワークを提案した。    
Fanら[47]は，画像パッチとフル画像の2つのソースからなるDual-Source Deep Convolutional Neural Network (DS-CNN)を提案しており，パッチに体の関節が含まれているかどうかを判定する関節検出と，パッチ内の関節の正確な位置を求める関節局在化の2つのタスクを行っている．
それぞれのタスクには損失関数が対応しており、2つのタスクを組み合わせることで、より良い結果が得られる。    
Luvizonら[48]は、2D/3Dポーズ推定とビデオシーケンスからのアクション認識を共同で処理するマルチタスクネットワークを学習した。  

### Body part detection methods 身体部位検出法
Body part detection methods for HPE aim to train a body part detector to predict the positions of body joints.     
Recent detection methods tackle pose estimation as a heatmap prediction problem.     
Concretely, the goal is to estimate K heatmaps {H1, H2, ..., HK} for a total of K keypoints.    
The pixel value Hi(x, y) in each keypoint heatmap indicates the probability that the keypoint lies in the position (x, y) (see Fig. 3 (b)).     
The target (or ground-truth) heatmap is generated by a 2D Gaussian centered at the ground-truth joint location [39] [53].      
Thus pose estimation networks are trained by minimizing the discrepancy (e.g., the Mean Squared-Error (MSE)) between the predicted heatmaps and target heatmaps.   
Compared with joint coordinates, heatmaps provide richer supervision information by preserving the spatial location information to facilitate the training of convolutional networks.     
Therefore, there is a recent growing interest in leveraging heatmaps to represent the joint locations and developing effective CNN architectures for HPE, e.g., [53] [54] [39] [55] [56] [38] [40] [57] [58] [59] [60] [61] [62] [63] [64].     
Tompson et al. [53] combined CNN-based body part detector with a part-based spatial-model into a unified learning framework for 2D HPE.     
Lifshitz et al. [55] proposed a CNN-based method for predicting the locations of joints.    
It incorporates the keypoints votes and joint probabilities to determine the human pose representation.     
Wei et al. [40] introduced a convolutional networks-based sequential framework named Convolutional Pose Machines (CPM) to predict the locations of key joints with multi-stage processing (the convolutional networks in each stage utilize the 2D belief maps generated from previous stages and produce the increasingly refined predictions of body part locations).     
Newell et al. [38] proposed an encoder-decoder network named ”stacked hourglass” (the encoder in this network squeezes features through bottleneck and then the decoder expands them) to repeat bottom-up and top-down processing with intermediate supervision.     
The stacked hourglass (SHG) network consists of consecutive steps of pooling and upsampling layers to capture information at every scale.     
Since then, complex variations of the SHG architecture were developed for HPE.      
Chu et al. [65] designed novel Hourglass Residual Units (HRUs), which extend the residual units with a side branch of filters with larger receptive fields, to capture features from various scales.    
Yang et al. [59] designed a multi-branch Pyramid Residual Module (PRM) to replace the residual unit in SHG, leading to enhanced invariance in scales of deep CNNs.      
With the emergence of Generative Adversarial Networks (GANs) [66], they are explored in HPE to generate biologically plausible pose configurations and to discriminate the predictions with high confidence from those with low confidence, which could infer the potential poses for the occluded body parts. 
Chen et al. [67] constructed a structureaware conditional adversarial network, named Adversarial PoseNet, which contains an hourglass network-based pose generator and two discriminators to discriminate against reasonable body poses from unreasonable ones. 
Chou et al. [68] built an adversarial learning-based network with two stacked hourglass networks sharing the same structure as discriminator and generator, respectively. 
The generator estimates the location of each joint, and the discriminator distinguishes the ground-truth heatmaps and predicted ones.     
Different from GANs-based methods that take HPE network as the generator and utilize the discriminator to provide supervision, Peng et al. [69] developed an adversarial data augmentation network to optimize data augmentation and network training by treating HPE network as a discriminator and using augmentation network as a generator to perform adversarial augmentations.     
Besides these efforts in effective network design for HPE, body structure information is also investigated to provide more and better supervision information for building HPE networks.      
Yang et al. [70] designed an end-to-end CNN framework for HPE, which is able to find hard negatives by incorporating the spatial and appearance consistency among human body parts.  
A structured feature-level learning framework was proposed in [71] for reasoning the correlations among human body joints in HPE, which captures richer information of human body joints and improves the learning results.     
Ke et al. [72] designed a multi-scale structure-aware neural network, which combines multi-scale supervision, multi-scale feature combination, structure-aware loss information scheme, and a keypoint masking training method to improve HPE models in complex scenarios.     
Tang et al. [73] built a hourglass-based supervision network, termed as Deeply Learned Compositional Model, to describe the complex and realistic relationships among body parts and learn the compositional pattern information (the orientation, scale and shape information of each body part) in human bodies.     
Tang and Wu [74] revealed that not all parts are related to each other, therefore introduced a Part-based Branches Network to learn representations specific to each part group rather than a shared representation for all parts.    
Human poses in video sequences are (3D) spatiotemporal signals. Therefore, modeling the spatio-temporal information is important for HPE from videos.     
Jain et al. [75] designed a two-branch CNN framework to incorporate both color and motion features within frame pairs to build an expressive temporal-spatial model in HPE.    
Pfister et al. [76] proposed a convolutional network that is able to utilize temporal context information from multiple frames by using optical flow to align predicted heatmaps from neighbouring frames.     
Different from the previous videobased methods which are computationally intensive, Luo et al. [60] introduced a recurrent structure for HPE with Long Short-Term Memory (LSTM) [77] to capture temporal geometric consistency and dependency from different frames.     
This method results in a faster speed in training the HPE network for videos. 
Zhang et al. [78] introduced a key frame proposal network for capturing spatial and temporal information from frames and a human pose interpolation module for efficient video-based pose estimation.

HPE用の身体部位検出手法は、身体の関節の位置を予測する身体部位検出器を学習することを目的としています。    
最近の検出手法では、ポーズ推定をヒートマップ予測問題として取り組んでいます。    
具体的には，K個のキーポイントに対して，K個のヒートマップ{H1, H2, ..., HK}を推定することを目的としています．   
各キーポイントのヒートマップの画素値Hi(x, y)は，そのキーポイントが位置(x, y)にある確率を示している（図3(b)参照）．    
ターゲット（またはグランドトゥルース）のヒートマップは，グランドトゥルースの関節位置を中心とした2次元ガウシアンによって生成される[39][53]．     
このように，姿勢推定ネットワークは，予測ヒートマップとターゲットヒートマップの間の不一致（例えば，MSE（Mean Squared-Error））を最小化することによって学習される．  
ヒートマップは，関節座標と比較して，空間的な位置情報を保持することで，より豊かな監視情報を提供し，畳み込みネットワークの学習を容易にします．    
そのため、最近ではヒートマップを活用してジョイント・ロケーションを表現し、HPEのための効果的なCNNアーキテクチャを開発することに関心が高まっています（例：[53] [54] [39] [55] [56] [38] [40] [57] [58] [59] [60] [61] [62] [63] [64]）。    
Tompsonら[53]は，CNNベースの身体部位検出器と部位ベースの空間モデルを組み合わせて，2D HPEのための統合学習フレームワークを開発した．    
Lifshitzら[55]は、関節の位置を予測するためのCNNベースの手法を提案した。   
これは、キーポイントの投票と関節の確率を組み込んで、人間のポーズ表現を決定するものである。    
Weiら[40]は、Convolutional Pose Machines（CPM）と呼ばれる畳み込みネットワークベースの逐次フレームワークを導入し、多段階処理で主要な関節の位置を予測している（各段階の畳み込みネットワークは、前の段階で生成された2D信念マップを利用し、体の部分の位置の予測を徐々に精緻化していく）。    
Newellら[38]は，「stacked hourglass」と名付けられたエンコーダ・デコーダネットワーク（このネットワークでは，エンコーダがボトルネックを通って特徴を絞り，デコーダがそれを拡大する）を提案し，中間的な監督のもとでボトムアップとトップダウンの処理を繰り返す．    
積み上げ式砂時計（SHG）ネットワークは、あらゆるスケールの情報を取り込むために、層のプーリングとアップサンプリングを連続して行うことで構成されている。    
その後、SHGアーキテクチャの複雑なバリエーションがHPE向けに開発されました。     
Chuら[65]は，様々なスケールの特徴を捉えるために，より大きな受容野を持つフィルタの側枝で残差ユニットを拡張する，新しい砂時計型残差ユニット（HRU）を設計した．   
Yangら[59]は、SHGの残差ユニットを置き換えるために、マルチブランチのPyramid Residual Module (PRM)を設計し、ディープCNNのスケールに対する不変性を高めた。     
Generative Adversarial Networks（GAN）[66]の登場により、生物学的に妥当なポーズ構成を生成し、信頼度の高い予測と低い予測を識別することで、隠蔽された身体部位の潜在的なポーズを推測することが可能になった。
Chenら[67]は、Adversarial PoseNetと名付けられた、構造を考慮した条件付き敵対的ネットワークを構築した。このネットワークには、砂時計ネットワークベースのポーズ生成器と2つの識別器が含まれており、合理的な体のポーズと不合理な体のポーズを識別することができる。
Chouら[68]は、同じ構造を持つ2つの砂時計型ネットワークを識別器と生成器として重ね合わせた、逆学習ベースのネットワークを構築した。
生成器は各関節の位置を推定し，弁別器は地表のヒートマップと予測されたヒートマップを区別する．    
Pengら[69]は、HPEネットワークを生成器とし、識別器を利用して監視を行うGANsベースの手法とは異なり、HPEネットワークを識別器として扱い、拡張ネットワークを生成器として利用して敵対的な拡張を行うことで、データ拡張とネットワークの学習を最適化する敵対的データ拡張ネットワークを開発した。    
HPEのための効果的なネットワーク設計におけるこれらの取り組みに加えて、HPEネットワークを構築するために、より多くの優れた監視情報を提供するために、身体構造情報も調査されている。     
Yangら[70]は、HPEのためのエンド・ツー・エンドのCNNフレームワークを設計しました。このフレームワークでは、人体のパーツ間の空間的および外観的な一貫性を組み込むことで、ハードネガティブを見つけることができます。  
71]では、HPEにおける人体の関節間の相関を推論するために、構造化された特徴レベルの学習フレームワークが提案されており、人体の関節のより豊かな情報を取り込み、学習結果を向上させている。     
Keら[72]は，複雑なシナリオにおけるHPEモデルを改善するために，マルチスケール監督，マルチスケール特徴の組み合わせ，構造を考慮した損失情報スキーム，およびキーポイントマスキング学習法を組み合わせた，マルチスケール構造考慮ニューラルネットワークを設計した．      
Tangら[73]は，Deeply Learned Compositional Modelと呼ばれる砂時計型のスーパービジョンネットワークを構築し，体の部位間の複雑で現実的な関係を記述し，人体の構成パターン情報（各体の部位の向き，スケール，形状の情報）を学習している．     
Tang and Wu [74]は、すべてのパーツが互いに関連しているわけではないことを明らかにし、したがって、すべてのパーツに共通の表現ではなく、各パーツグループに固有の表現を学習するために、Part-based Branches Networkを導入した。    
ビデオシーケンス中の人間のポーズは、（3D）時空間信号です。したがって、時空間情報をモデル化することは、ビデオからのHPEにとって重要である。    
Jainら[75]は、HPEにおいて表現力のある時間-空間モデルを構築するために、フレームペア内の色と動きの両方の特徴を組み込む2ブランチCNNフレームワークを設計した。   
Pfisterら[76]は，オプティカルフローを用いて隣接するフレームからの予測ヒートマップを揃えることで，複数のフレームからの時間的コンテキスト情報を利用できる畳み込みネットワークを提案した．    
Luoら[60]は，計算量の多いこれまでのビデオベースの手法とは異なり，異なるフレームからの時間的な幾何学的整合性と依存性を捉えるために，長短期記憶（LSTM）[77]を用いたHPE用のリカレント構造を導入した．    
この方法では，動画に対するHPEネットワークの学習が高速化される．
Zhangら[78]は，フレームから空間的・時間的情報を取得するためのキーフレーム提案ネットワークと，動画ベースの姿勢推定を効率的に行うための人物姿勢補間モジュールを導入した．

### 3.2 2D multi-person pose estimation 2D多人数のポーズ推定
Compared to single-person HPE, multi-person HPE is more difficult and challenging because it needs to figure out the number of people and their positions, and how to group keypoints for different people.    
In order to solve these problems, multi-person HPE methods can be classified into top-down and bottom-up methods.    
Top-down methods employ off-the-shelf person detectors to obtain a set of boxes (each corresponding to one person) from the input images, and then apply single-person pose estimators to each person box to generate multi-person poses.     
Different from top-down methods, bottom-up methods locate all the body joints in one image first and then group them to the corresponding subjects.     
In the top-down pipeline, the number of people in the input image will directly affect the computing time.     
The computing speed for bottom-up methods is usually faster than top-down methods since they do not need to detect the pose for each person separately. Fig. 4 shows the general frameworks for 2D multi-person HPE methods.  

1人用のHPEに比べて、多人数用のHPEは、人数とその位置を把握する必要があり、また、異なる人のためのキーポイントをどのようにグループ化するかという問題があり、より難しく、困難です。    
これらの問題を解決するために、多人数HPEの手法は、トップダウン方式とボトムアップ方式に分類されます。   
 トップダウン方式は、市販の人物検出器を用いて、入力画像から1人に対応するボックスの集合を得て、各人物ボックスに1人用のポーズ推定器を適用して、複数人用のポーズを生成する方式である。    
トップダウン方式とは異なり、ボトムアップ方式では、まず1つの画像内のすべての体の関節を見つけ、次にそれらを対応する被験者にグループ化する。    
トップダウン方式では、入力画像に含まれる人物の数が計算時間に直接影響します。    
ボトムアップ方式では、各人のポーズを個別に検出する必要がないため、通常、トップダウン方式よりも計算速度が速くなる。図4は、2D多人数HPE法の一般的なフレームワークを示している。  

#### 3.2.1 Top-down pipeline トップダウンパイプライン
In the top-down pipeline as shown in Fig. 4 (a), there are two important parts: a human body detector to obtain person bounding boxes and a single-person pose estimator to predict the locations of keypoints within these bounding boxes.   
A line of works focus on designing and improving the modules in HPE networks, e.g., [79] [80] [62] [81] [82] [83] [84] [85] [86] [87].     
For example, in order to answer the question ”how good could a simple method be” in building an HPE network, Xiao et al. [62] added a few deconvolutional layers in the ResNet (backbone network) to build a simple yet effective structure to produce heatmaps for high-resolution representations.     
Sun et al. [81] presented a novel High-Resolution Net (HRNet) to learn reliable high resolution representations by connecting mutli-resolution subnetworks in parallel and conducting repeated multi-scale fusions.     
To improve the accuracy of keypoint localization, Wang et al. [84] introduced a two-stage graph-based and model-agnostic framework, called GraphPCNN. It consists of a localization subnet to obtain rough keypoint locations and a graph pose refinement module to get refined keypoints localization representations.      
In order to obtain more precise keypoints localization, Cai et al. [86] introduced a multi-stage network with a Residual Steps Network (RSN) module to learn delicate local representations by efficient intra-level feature fusion strategies, and a Pose Refine Machine (PRM) module to find a trade-off between local and global representations in the features.     
Estimating poses under occlusion and truncation scenes often occurs in multi-person settings since the overlapping of limbs is inevitable.      
Human detectors may fail in the first step of top-down pipeline due to occlusion or truncation. Therefore, robustness to occlusion or truncation is an important aspect of the multi-person HPE approaches.      
Towards this goal, Iqbal and Gall [88] built a convolutional pose machinebased pose estimator to estimate the joint candidates.      
Then they used integer linear programming (ILP) to solve the joint-to-person association problem and obtain the human body poses even in presence of severe occlusions.      
Fang et al. [89] designed a novel regional multi-person pose estimation (RMPE) approach to improve the performance of HPE in complex scenes.      
Specifically, RMPE framework has three parts: Symmetric Spatial Transformer Network (to detect single person region within inaccurate bounding box), Parametric Pose Non-Maximum-Suppression (to solve the redundant detection problem), and Pose-Guided Proposals Generator (to augment training data). 
Papandreou et al. [79] proposed a two-stage architecture, consisting of a Faster R-CNN person detector to create bounding boxes for candidate human bodies and a keypoint estimator to predict the locations of keypoints by using a form of heatmap-offset aggregation.     
The overall method works well in occluded and cluttered scenes.     
In order to alleviate the occlusion problem in HPE, Chen et al. [90] presented a Cascade Pyramid Network (CPN) which includes two parts: GlobalNet (a feature pyramid network to predict the invisible keypoints like eyes or hands) and RefineNet (a network to integrate all levels of features from the GlobalNet with a keypoint mining loss).     
Their results reveal that CPN has a good performance in predicting occluded keypoints.     
Su et al. [91] designed two modules, the Channel Shuffle Module and the Spatial & Channel-wise Attention Residual Bottleneck, to achieve channel-wise and spatial information enhancement for better multi-person pose estimation under occluded scenes.      
Qiu et al. [92] developed an Occluded Pose Estimation and Correction (OPEC-Net) module and an occluded pose dataset to solve the occlusion problem in crowd pose estimation.      
Umer et al. [93] proposed a keypoint correspondence framework to recover missed poses using temporal information of the previous frame in occluded scene.      
The network is trained using self-supervision in order to improve the pose estimation results in sparsely annotated video datasets.   

図4(a)に示すように、トップダウン・パイプラインには、2つの重要な部分がある。すなわち、人物のバウンディング・ボックスを得るための人体検出器と、このバウンディング・ボックス内のキーポイントの位置を予測するための一人用ポーズ推定器である。    
例えば、[79] [80] [62] [81] [82] [83] [84] [85] [86] [87]のように、HPEネットワークのモジュールの設計と改善に焦点を当てた研究があります。   
例えば、Xiaoら[62]は、HPEネットワークを構築する際に「シンプルな手法でどれだけの効果が得られるか」という疑問に答えるために、ResNet（バックボーンネットワーク）にいくつかのデコンボリューション層を追加し、高解像度表現のヒートマップを生成するためのシンプルかつ効果的な構造を構築しました。    
Sunら[81]は、複数の解像度のサブネットワークを並列に接続し、マルチスケールフュージョンを繰り返し行うことで、信頼性の高い高解像度表現を学習する、新しい高解像度ネット（HRNet）を発表した。    
キーポイントの定位精度を向上させるために、Wangら[84]は、GraphPCNNと呼ばれる2段階のグラフベースでモデル非依存のフレームワークを導入した。このフレームワークは、大まかなキーポイントの位置を得るための局在化サブネットと、洗練されたキーポイントの局在化表現を得るためのグラフポーズ精密化モジュールで構成されている。     
Caiら[86]は、より正確なキーポイントの定位を得るために、効率的なイントラレベルの特徴融合戦略によって繊細なローカル表現を学習するResidual Steps Network（RSN）モジュールと、特徴におけるローカル表現とグローバル表現の間のトレードオフを見つけるPose Refine Machine（PRM）モジュールを備えたマルチステージ・ネットワークを導入した。    
オクルージョンやトランケーションのあるシーンでのポーズの推定は、手足の重なりが避けられないため、多人数の設定でよく起こります。     
人間の検出器は、トップダウンパイプラインの最初のステップで、オクルージョンやトランケーションのために失敗することがあります。したがって、オクルージョンやトランケーションに対するロバスト性は、多人数向けHPEアプローチの重要な側面です。     
この目標に向けて、Iqbal and Gall [88]は、関節候補を推定するために、畳み込みポーズマシンベースのポーズ推定器を構築した。     
そして、整数線形計画法（ILP）を用いて、関節と人の関連付け問題を解決し、深刻なオクルージョンがある場合でも、人体のポーズを得ることができた。     
Fangら[89]は、複雑なシーンでのHPEの性能を向上させるために、新しい地域別多人数ポーズ推定（Regional Multi-Person Pose Estimation: RMPE）アプローチを設計しました。     
具体的には、RMPEフレームワークには3つの部分があります。具体的には、RMPEフレームワークは、Symmetric Spatial Transformer Network（不正確なバウンディングボックス内の一人の人物領域を検出する）、Parametric Pose Non-Maximum-Suppression（冗長な検出問題を解決する）、Pose-Guided Proposals Generator（トレーニングデータを補強する）の3つの部分から構成されている。
Papandreouら[79]は、候補となる人体のバウンディングボックスを作成するFaster R-CNN人物検出器と、ヒートマップ・オフセット集計の形式を用いてキーポイントの位置を予測するキーポイント推定器からなる2段階のアーキテクチャを提案している。     
この手法は，オクルージョンのあるシーンや散乱したシーンでもうまく機能します．    
HPEにおけるオクルージョンの問題を軽減するために、Chenら[90]はCascade Pyramid Network (CPN)を発表した。GlobalNet（目や手のような目に見えないキーポイントを予測するための特徴ピラミッドネットワーク）とRefineNet（キーポイントのマイニング損失でGlobalNetからのすべてのレベルの特徴を統合するネットワーク）の2つの部分を含むCascade Pyramid Network（CPN）を発表しました。    
彼らの結果から、CPNはオクルージョンしたキーポイントを予測するのに良い性能を持っていることが明らかになった。    
Suら[91]は、チャネルシャッフルモジュールと空間的＆チャネル的注意の残留ボトルネックという2つのモジュールを設計し、チャネル的および空間的な情報強化を実現して、オクルードシーン下でのより優れた複数人のポーズ推定を実現しています。     
Qiuら[92]は、群集のポーズ推定におけるオクルージョン問題を解決するために、Occluded Pose Estimation and Correction (OPEC-Net)モジュールとオクルージョン・ポーズ・データセットを開発しました。     
Umerら[93]は、隠蔽されたシーンで前のフレームの時間情報を使ってミスポーズを回復するキーポイント対応フレームワークを提案した。     
このネットワークは、アノテーションの少ないビデオデータセットにおけるポーズ推定結果を改善するために、自己教師法を用いて学習される。

#### 3.2.2 Bottom-up pipeline ボトムアップ・パイプライン 
The bottom-up pipeline (e.g., [94] [95] [96] [17] [97] [98] [99] [100] [101] [102] [103]) has two main steps including body joint detection (i.e., extracting local features and predicting human body joint candidates) and joint candidates assembling for individual bodies (i.e., grouping joint candidates to build final pose representations with part association strategies) as illustrated in Fig. 4 (b).    
Pishchulin et al. [94] proposed a Fast R-CNN based body part detector named DeepCut, which is one of the earliest two-step bottom-up approaches.    
It first detects all the body part candidates, then labels each part and assembles these
parts using integer linear programming (ILP) to a final pose.    
However, DeepCut model is computationally expensive.    
To this end, Insafutdinov et al. [95] introduced DeeperCut to improve DeepCut by applying a stronger body part detector with a better incremental optimization strategy and imageconditioned pairwise terms to group body parts, leading to improved performance as well as a faster speed.    
Later, Cao et al. [17] built a detector named OpenPose, which uses Convolutional Pose Machines [40] (CPMs) to predict keypoints coordinates via heatmaps and Part Affinity Fields (PAFs, a set of 2D vector fields with vectormaps that encode the position and orientation of limbs) to associate the keypoints to each person.     
OpenPose largely accelerates the speed of the bottomup multi-person HPE.    
Based on the OpenPose framework, Zhu et al. [104] improved the OpenPose structure by adding redundant edges to increase the connections between joints in PAFs and obtained better performance than the baseline approach.      
Although OpenPose-based methods have achieved impressive results on high resolution images, they have poor performance with low resolution images and occlusions.   
To address this problem, Kreiss et al. [100] proposed a bottomup method called PifPaf that uses a Part Intensity Field (PIF) to predict the locations of body parts and a Part Association Field (PAF) to represent the joints association.        
This method outperformed previous OpenPose-based approaches on low resolution and occluded scenes.      
Motivated by OpenPose [17] and stacked hourglass structure [38], Newell et al. [97] introduced a single-stage deep network to simultaneously obtain pose detections and group assignments.    
Following [97], Jin et al. [102] proposed a new differentiable Hierarchical Graph Grouping (HGG) method to learn the human part grouping.      
Based on [97] and [81], Cheng et al. [103] proposed a simple extension of HRNet, named Higher Resolution Network (HigherHRNet), which deconvolves the high-resolution heatmaps generated by HRNet to solve the scale variation challenge in bottom-up multi-person pose estimation.    
Multi-task structures are also employed in bottom-up HPE methods.      
Papandreou et al. [105] introduced PersonLab to combine the pose estimation module and the person segmentation module for keypoints detection and association.     
PersonLab consists of short-range offsets (for refining heatmaps), mid-range offsets (for predicting the keypoints) and long-range offsets (for grouping keypoints into instances).      
Kocabas et al. [106] presented a multi-task learning model with a pose residual net, named MultiPoseNet, which can perform keypoints prediction, human detection and semantic segmentation tasks altogether.     

ボトムアップ・パイプライン（例えば、[94] [95] [96] [17] [97] [98] [99] [100] [101] [102] [103]）には、図4（b）に示すように、体の関節の検出（すなわち、局所的な特徴を抽出し、人体の関節候補を予測すること）と、個々の体に対する関節候補の組み立て（すなわち、関節候補をグループ化し、パーツの関連付け戦略を用いて最終的なポーズ表現を構築すること）の2つの主要なステップがある。   
Pishchulinら[94]は，DeepCutと名付けたR-CNNベースの高速身体部位検出器を提案しているが，これは初期の2段階ボトムアップアプローチの1つである．   
DeepCutは、最初にすべてのボディパーツ候補を検出し、次に各パーツにラベルを付け、整数線形計画（ILP）を用いてこれらのパーツを組み立てます。
整数線形計画法（ILP）を用いて、これらのパーツを組み立て、最終的なポーズを決定します。   
しかし、DeepCutモデルは計算量が多い。   
このため、Insafutdinovら[95]は、DeepCutを改良するためにDeeperCutを導入した。このDeeperCutは、より強力な身体部位検出器と、より優れた漸進的最適化戦略、および身体部位をグループ化するための画像条件付きペアワイズ用語を適用することで、性能の向上と高速化を実現している。 
その後，Caoら[17]は，Convolutional Pose Machines [40]（CPM）を用いてヒートマップによりキーポイントの座標を予測し，Part Affinity Fields（PAF，手足の位置と向きをエンコードするベクターマップを持つ2次元ベクトルフィールドの集合）を用いてキーポイントを各人物に関連付けるOpenPoseという検出器を構築した．     
OpenPoseは、ボトムアップ型の多人数HPEのスピードを大幅に向上させます。   
Zhuら[104]は，OpenPoseフレームワークに基づいて，PAFの関節間の接続を増やすために冗長なエッジを追加してOpenPose構造を改良し，ベースラインアプローチよりも優れた性能を得た．     
OpenPoseベースの手法は、高解像度の画像において 高解像度の画像では素晴らしい結果が得られますが，低解像度の画像や OpenPoseを用いた手法は，高解像度の画像では素晴らしい結果を得ることができますが，低解像度の画像やオクルージョンに対しては性能が劣ります．    
この問題を解決するために，Kreissら[100]は，体のパーツの位置を予測するためにPart Intensity Field（PIF）を使用し，関節の関連性を表現するためにPart Association Field（PAF）を使用するPifPafというボトムアップ手法を提案した．    
この手法は，低解像度のシーンや隠蔽されたシーンにおいて，これまでのOpenPoseベースのアプローチよりも優れた性能を発揮した．     
OpenPose[17]やstacked hourglass structure[38]に触発されて，Newellら[97]は，シングルステージのディープネットワークを導入して，ポーズの検出とグループの割り当てを同時に行った．    
Jinら[102]は、[97]に続いて、人間のパーツのグルーピングを学習するために、新しい微分可能なHierarchical Graph Grouping (HGG)法を提案しました。     
Chengら[103]は、[97]と[81]に基づいて、HRNetの単純な拡張であるHigher Resolution Network (HigherHRNet)を提案した。この拡張は、HRNetによって生成された高解像度ヒートマップを分解し、ボトムアップ式の多人数ポーズ推定におけるスケール変動の問題を解決する。   
ボトムアップHPE法では、マルチタスク構造も採用されている。     
Papandreouら[105]は、PersonLabを導入し、ポーズ推定モジュールと人物セグメンテーションモジュールを組み合わせて、キーポイントの検出と関連付けを行っている。      
PersonLab は、短距離オフセット（ヒートマップの改良のため）、中距離オフセット（キーポイントの予測のため）、長距離オフセット（キーポイントをインスタンスにグループ化するため）から構成されている。     
Kocabasら[106]は、MultiPoseNetと名付けたポーズ残差ネットを用いたマルチタスク学習モデルを発表した。このモデルは、キーポイントの予測、人物の検出、セマンティックセグメンテーションのタスクをすべて実行できる。 

### 3.3 2D HPE Summary 2D HPEのまとめ
In summary, the performance of 2D HPE has been significantly improved with the blooming of deep learning techniques.     
In recent years, deeper and more powerful networks have promoted the performance in 2D singleperson HPE such as DeepPose [36] and Stacked Hourglass Network [38], as well as in 2D multi-person HPE such as AlphaPose [89] and OpenPose [17].    
Although these works have achieved sufficiently good performance in different 2D HPE scenarios, problems still remain.      
Regression and body part detection methods have their own advantages and limitations in 2D single-person HPE.      
Regression methods can learn a nonlinear mapping from input images to keypoint coordinates with an end-toend framework, which offer a fast learning paradigm and a sub-pixel level prediction accuracy.       
However, they usually give sub-optimal solutions [44] due to the highly nonlinear problem.      
Body part detection methods, in particular heatmapbased frameworks, are more widely used in 2D HPE since     
(1) the probabilistic prediction of each pixel in heatmap can improve the accuracy of locating the keypoints; and (2) heatmaps provide richer supervision information by preserving the spatial location information.    
However, the precision of the predicted keypoints is dependent on the resolution of heatmaps.     
The computational cost and memory footprint are significantly increased when using high resolution heatmaps.    
As for the top-down and bottom-up pipelines for 2D multi-person HPE, it is difficult to identify which method is better since both of them are widely used in recent works
with their strengths and weaknesses.     
On one hand, topdown pipeline yields better results because it first detects each individual from the image using detection methods, then predicts the locations of keypoints using the single person-based approaches.     
In this case, the keypoint heatmap estimation within each detected person region is eased as the background is largely removed.     
On the other hand, bottomup methods are generally faster than top-down methods, because they directly detect all the keypoints and group them into individual poses using keypoint association strategies such as affinity linking [17], associative embedding [97], and pixel-wise keypoint regression [107].      
There are several challenges in 2D HPE which need to be further addressed in future research.     
First is the reliable detection of individuals under significant occlusion, e.g., in crowd scenarios.     
The person detectors in top-down 2D HPE methods may fail to identify the boundaries of largely overlapped human bodies.        
Similarly, the difficulty of keypoint association is more pronounced for bottom-up approaches in occluded scenes.     
The second challenge is computation efficiency.      
Although some methods like OpenPose [17] can achieve near real-time processing on special hardware with moderate computing power (e.g., 22 FPS on a machine with a Nvidia GTX 1080 Ti GPU), it is still difficult to implement the networks on resource-constrained devices.      
Real-world applications (e.g., online coaching, gaming, AR and VR) require more efficient HPE methods on commercial devices which can bring better interaction experience for users.    
Another challenge lies in the limited data for rare poses.     
Although the size of current datasets for 2D HPE is large enough (e.g., COCO dataset [108]) for the normal pose estimation (e.g., standing, walking, running), these datasets have limited training data for unusual poses, e.g., falling.     
The data imbalance may cause model bias, resulting in poor performance on those poses. It would be useful to develop effective data generation or augmentation techniques to generate extra pose data for training more robust models.    

まとめると、2D HPEの性能は、深層学習技術の開花によって大幅に向上した。    
近年、より深く、より強力なネットワークが、DeepPose [36]やStacked Hourglass Network [38]などの2D一人用HPEや、AlphaPose [89]やOpenPose [17]などの2D複数人用HPEの性能を向上させました。AlphaPose[89]やOpenPose[17]などの2D多人数HPEでも性能が向上しました。   
これらの作品は、さまざまな2D HPEシナリオにおいて十分に優れた性能を達成しているが、問題はまだ残っている。     
回帰法と身体部位検出法は、2Dの1人用HPEにおいて、それぞれ利点と限界があります。     
回帰法は、入力画像からキーポイント座標への非線形マッピングをエンド・ツー・エンドのフレームワークで学習することができ、高速な学習パラダイムとサブピクセルレベルの予測精度を提供します。      
しかし，非線形性の高い問題であるため，通常は最適ではない解を与えてしまいます[44]．     
2D HPEでは、身体部位検出法、特にヒートマップベースのフレームワークがより広く使用されていますが、その理由は以下の通りです。    
(1) ヒートマップの各ピクセルの確率的な予測は，キーポイントの位置を特定する精度を向上させることができ，(2) ヒートマップは，空間的な位置情報を保持することで，より豊かな監視情報を提供する．   
しかし，予測されるキーポイントの精度は，ヒートマップの解像度に依存する．    
高解像度のヒートマップを使用すると，計算コストとメモリフットプリントが大幅に増加する．   
トップダウン型とボトムアップ型の2つの手法は，それぞれ長所と短所を持ちながらも，最近の研究で広く用いられているため，どちらが優れているかを特定するのは難しい．
どちらの手法が優れているかを特定することは困難ですが、両者はそれぞれ長所と短所を持っています。    
一方、トップダウンパイプラインでは、まず画像から各個人を検出法で検出し、次に一人ベースのアプローチでキーポイントの位置を予測するため、より良い結果が得られます。    
この場合、検出された各人物領域内のキーポイント・ヒートマップの推定は、背景が大幅に除去されるため、容易になります。    
一方，ボトムアップ法は，親和性リンク[17]，連想埋め込み[97]，ピクセル単位のキーポイント回帰[107]などのキーポイント関連戦略を用いて，すべてのキーポイントを直接検出し，個々のポーズにグループ化するため，一般にトップダウン法よりも高速である．     
2D HPEには、今後の研究でさらに解決しなければならないいくつかの課題がある。    
1つ目の課題は、群衆の中などの大きなオクルージョンの下で、個人を確実に検出することである。    
トップダウン式の2D HPE手法の人物検出器は、大きく重なった人体の境界を識別できないことがある。       
同様に、オクルージョンがあるシーンでは、ボトムアップ方式ではキーポイントの関連付けが難しくなります。    
2つ目の課題は、計算効率です。     
OpenPose[17]のように、中程度の計算能力を持つ特殊なハードウェアでほぼリアルタイム処理を実現できる手法もあるが（例：Nvidia GTX 1080 Ti GPUを搭載したマシンで22 FPS）、リソースに制約のあるデバイスにネットワークを実装することは依然として困難である。     
現実世界のアプリケーション（例：オンラインコーチング、ゲーム、ARやVR）では、ユーザーにより良いインタラクション体験をもたらすことができる、商用デバイス上でのより効率的なHPE手法が必要です。   
もう一つの課題は、珍しいポーズのデータが限られていることです。    
2D HPE用の現在のデータセットは、通常のポーズ推定（例：立っている、歩いている、走っている）には十分なサイズ（例：COCOデータセット[108]）であるが、これらのデータセットでは、落下などの珍しいポーズのトレーニングデータが限られている。    
このようなデータの不均衡がモデルの偏りを引き起こし、結果的にこれらのポーズでのパフォーマンスが低下する可能性がある。よりロバストなモデルを学習するために、追加のポーズデータを生成するための効果的なデータ生成または増強技術を開発することは有益である。   

## 4 3D HUMAN POSE ESTIMATION 3D人体姿勢推定
3D HPE, which aims to predict locations of body joints in 3D space, has attracted much interest in recent years since it can provide extensive 3D structure information related to the human body.      
It can be applied to various applications (e.g., 3D movie and animation industries, virtual reality, and online 3D action prediction).       
Although significant improvements have recently been achieved in 2D HPE, 3D HPE still remains as a challenging task.     
Most existing research works tackle 3D HPE from monocular images or videos, which is an ill-posed and inverse problem due to projection of 3D to 2D where one dimension is lost.     
When multiple viewpoints are available or other sensors such as IMU and LiDAR are deployed, 3D HPE can be a well-posed problem employing information fusion techniques.     
Another limitation is that deep learning models are data-hungry and sensitive to the data collection environment.     
Unlike 2D human datasets where accurate 2D pose annotation can be easily obtained, collecting accurate 3D pose annotation is time-consuming and manual labeling is not practical.     
Also, datasets are usually collected from indoor environments with selected daily actions.     
Recent works [109] [110] [111] have validated the poor generalization of models trained with biased datasets by cross-dataset inference [112].    
In this section, we first focus on 3D HPE from monocular RGB images and videos, and then cover 3D HPE based on other types of sensors.  

3D HPEは、3D空間における体の関節の位置を予測することを目的としており、人体に関する広範な3D構造情報を提供できることから、近年注目を集めています。     
3D HPEは、3D映画やアニメーション、バーチャルリアリティ、オンライン3Dアクション予測など、様々な用途に応用することができます。      
最近、2D HPEでは大幅な改善が達成されましたが、3D HPEはまだ困難な課題として残っています。    
既存の研究では、単眼画像やビデオからの3D HPEに取り組んでいますが、これは、3Dから2Dへの投影により1つの次元が失われるため、逆問題となります。    
複数の視点が利用可能であったり、IMUやLiDARなどの他のセンサーが配備されている場合、3D HPEは情報融合技術を用いた正解の問題となります。    
もう一つの限界は、深層学習モデルがデータを必要とし、データ収集環境に敏感であることです。    
正確な2Dポーズアノテーションが簡単に得られる2Dヒューマンデータセットとは異なり、正確な3Dポーズアノテーションの収集には時間がかかり、手動でのラベリングは現実的ではありません。    
また、データセットは通常、選択された日常動作を伴う屋内環境から収集される。    
最近の研究 [109] [110] [111]では、クロスデータセット推論によって、偏ったデータセットで訓練されたモデルの一般化が不十分であることが検証されている [112]。   
このセクションでは、まず、単眼のRGB画像やビデオからの3D HPEに焦点を当て、次に、他のタイプのセンサーに基づく3D HPEをカバーする。

### 4.1 3D HPE from monocular RGB images and videos 単眼のRGB画像・動画からの3D HPE
The monocular camera is the most widely used sensor for HPE in both 2D and 3D scenarios.    
Recent progress of deep learning-based 2D HPE from monocular images and videos has enabled researchers to extend their works to 3D HPE.    
Specifically, deep learning-based 3D HPE methods are divided into two broad categories: single-view 3D HPE and multi-view 3D HPE.  

単眼カメラは、2Dと3Dの両方のシナリオでHPEに最も広く使用されているセンサーです。   
最近の深層学習を用いた単眼画像や動画からの2D HPEの進展により、研究者たちはその研究を3D HPEにも拡張することができるようになった。   
具体的には、深層学習ベースの3D HPE手法は、単視点3D HPEと多視点3D HPEの2つに大別される。  

#### 4.1.1 Single-view 3D HPE 単眼の3D HPE
The reconstruction of 3D human poses from a single view of monocular images and videos is a nontrivial task that suffers from self-occlusions and other object occlusions, depth ambiguities, and insufficient training data.     
It is a severely ill-posed problem because different 3D human poses can be projected to a similar 2D pose projection.    
Moreover, for methods that build upon 2D joints, minor localization errors of the 2D body joints can lead to large pose distortion in the 3D space.    
Compared to the single-person scenario, the multi-person case is more complicated. Thus they are discussed separately in what follows.    
Single-person 3D HPE approaches can be classified into model-free and model-based categories based on whether they employ a human body model (as listed in Section 2) to estimate 3D human pose or not. 

単眼画像やビデオの単一ビューから人間の3Dポーズを再構築することは、自己包摂や他のオブジェクトのオクルージョン、深度の曖昧さ、不十分なトレーニングデータなどの問題に悩まされる非自明なタスクです。    
また、異なる3次元の人間のポーズは、類似した2次元のポーズ投影に投影されるため、非常に難しい問題です。   
さらに、2次元の関節を利用する手法では、2次元の体の関節のわずかな位置の誤差が、3次元空間での大きなポーズの歪みにつながります。   
一人の場合に比べて、複数人の場合はより複雑になります。そのため、以下では別々に説明します。   
1人用の3D HPEアプローチは、人間の3Dポーズを推定するために人体モデル（セクション2に記載）を使用するかどうかに基づいて、モデルフリーとモデルベースのカテゴリーに分類されます。  

##### Fig. 5: Single-person 3D HPE frameworks.    一人用の3D HPEフレームワーク。
(a) Direct estimation approaches directly estimate the 3D human pose from 2D images.    
(b) 2D to 3D lifting approaches leverage the predicted 2D human pose (intermediate representation) for 3D pose estimation.   
(c) Model-based methods incorporate parametric body models to recover high-quality 3D human mesh.      
The 3D pose and shape parameters inferred by the 3D pose and shape network are fed into the model regressor to reconstruct 3D human mesh.    
Part of the figure is from [113].   

(a) 直接推定アプローチは、2D画像から人間の3Dポーズを直接推定する。   
(b) 2Dから3Dへのリフティングアプローチは、予測された2Dの人間のポーズ（中間表現）を3Dポーズの推定に利用する。  
(c) モデルベース手法では、パラメトリックなボディモデルを用いて、高品質な3D人体メッシュを復元する。    
3Dポーズ・形状ネットワークによって推論された3Dポーズ・形状パラメータは、モデル回帰器に与えられ、3Dヒューマンメッシュを復元する。    
図の一部は[113]より引用。  

#### Model-free methods.    モデルフリー方式。
The model-free methods do not employ human body models to reconstruct 3D human representation.     
These methods can be further divided into two classes: (1) Direct estimation approaches, and (2) 2D to 3D lifting approaches.    
Direct estimation: As shown in Fig. 5(a), direct estimation methods infer the 3D human pose from 2D images without intermediately estimating 2D pose representation, e.g., [114] [115] [116] [43] [117] [118] [119].    
One of the early deep learning approaches was proposed by Li and Chan [114].    
They employed a shallow network to train the body part detection with sliding windows and the pose coordinate regression synchronously.     
A follow-up approach was proposed by Li et al. [115] where the image-3D pose pairs were used as the network input.    
The score network can assign high scores to the correct image-3D pose pairs and low scores to other pairs.     
However, these approaches are highly inefficient because they require multiple forward network inferences.       
Sun et al. [43] proposed a structure-aware regression approach. Instead of using joint-based representation, they adopted a bonebased representation with more stability. 
A compositional loss was defined by exploiting 3D bone structure with bonebased representation that encodes long range interactions between the bones.    
Tekin et al. [116] encoded the structural dependencies between joints by learning a mapping of 3D pose to a high-dimensional latent space.     
The learned highdimensional pose representation can enforce structural constraints of the 3D pose.    
Pavlakos et al. [117] [118] introduced  a volumetric representation to convert the highly non-linear 3D coordinate regression problem to a manageable form in a discretized space.     
The voxel likelihoods for each joint in the volume were predicted by a convolutional network.      
Ordinal depth relations of human joints were used to alleviate the need for accurate 3D ground truth pose.   

モデルフリー手法とは、人体モデルを使用せずに、人間の3D表現を再構築する手法です。    
これらの手法は、さらに2つのクラスに分けられます。(1) 直接的な推定アプローチと、(2) 2Dから3Dへのリフティングアプローチです。   
直接推定。図5（a）に示すように、直接推定法は、2次元のポーズ表現を間欠的に推定することなく、2次元の画像から3次元の人間のポーズを推論するもので、例えば、[114] [115] [116] [43] [117] [118] [119]などがある。   
初期の深層学習アプローチの1つは，Li and Chan [114]によって提案されたものである．   
彼らは、浅いネットワークを用いて、スライディングウィンドウを用いた身体部位検出と、ポーズ座標回帰を同期して学習させた。    
その後、Liら[115]により、画像と3Dポーズのペアをネットワークの入力として使用するアプローチが提案された。   
スコアネットワークは，正しい画像-3Dポーズのペアには高いスコアを，その他のペアには低いスコアを割り当てることができる．    
しかし、これらのアプローチは、複数の前方ネットワーク推論を必要とするため、非常に非効率的である。  
Sunら[43]は、構造を考慮した回帰アプローチを提案した。彼らは、関節ベースの表現の代わりに、より安定性の高い骨ベースの表現を採用しました。   
骨間の長距離相互作用を符号化する骨ベースの表現を用いて3次元の骨構造を利用することで、組成損失を定義した。   
Tekinら[116]は、3Dポーズの高次元潜在空間へのマッピングを学習することで、関節間の構造依存性を符号化した。    
学習された高次元のポーズ表現は、3Dポーズの構造的制約を強化することができる。   
Pavlakosら[117][118]は、高度に非線形な3次元座標回帰問題を離散化された空間で管理可能な形に変換するために、ボリューム表現を導入した。    
ボリューム内の各関節のボクセル尤度は、畳み込みネットワークによって予測された。     
人間の関節の深度関係を利用して、正確な3Dグランドトゥルースポーズの必要性を軽減した。

#### 2D to 3D lifting: 2D→3Dリフティング 
Motivated by the recent success of 2D HPE, 2D to 3D lifting approaches that infer 3D human pose from the intermediately estimated 2D human pose have become a popular 3D HPE solution as illustrated in Fig. 5(b).   
Benefiting from the excellent performance of state-of-the-art 2D pose detectors, 2D to 3D lifting approaches generally outperform direct estimation approaches.     
In the first stage, off-the-shelf 2D HPE models are employed to estimate 2D pose, and then in the second stage 2D to 3D lifting is used to obtain 3D pose.     
Chen and Ramanan [120] deployed a nearest neighbor matching of the predicted 2D pose and 3D pose from a library.     
However, 3D HPE could fail when the 3D pose is not conditionally independent of the image given the 2D pose.       
Martinez et al. [121] proposed a simple but effective fully connected residual network to regress 3D joint locations based on the 2D joint locations.      
Despite achieving the state-of-the-art results at that time, the method could fail due to reconstruction ambiguity of over-reliance on the 2D pose detector [118].     
Tekin et al. [122] and Zhou et al. [123] utilized 2D heatmaps instead of 2D pose as intermediate representations for estimating 3D pose.     
Moreno-Noguer [124] inferred the 3D human pose via distance matrix regression where the distances of 2D and 3D body joints were encoded into two Euclidean Distance Matrices (EDMs).     
EDMs are invariant to in-plane image rotations and translations, as well as scaling invariance when applying normalization operations.     
Wang et al. [125] developed a Pairwise Ranking Convolutional Neural Network (PRCNN) to predict the depth ranking of pairwise human joints.     
Then, a coarseto-fine pose estimator was used to regress the 3D pose from 2D joints and the depth ranking matrix.    
Jahangiri and Yuille [126], Sharma et al. [127], and Li and Lee [128] first generated multiple diverse 3D pose hypotheses then applied ranking networks to select the best 3D pose.    

2D HPEの成功を受けて、図5（b）に示すように、間欠的に推定された2Dの人物姿勢から3Dの人物姿勢を推論する2D to 3Dリフティングアプローチが、3D HPEのソリューションとして一般的になってきた。  
最先端の2Dポーズ検出器の優れた性能を利用して、2D to 3Dリフティングアプローチは、一般的に直接推定アプローチよりも優れています。     
第1段階では、既製の2D HPEモデルを使用して2Dポーズを推定し、第2段階では、2D→3Dリフティングを使用して3Dポーズを取得する。    
Chen and Ramanan [120]は、予測された2Dポーズとライブラリからの3Dポーズの最近接マッチングを導入した。    
しかし、3Dポーズが2Dポーズを与えられた画像から条件付きで独立していない場合、3D HPEは失敗する可能性がある。      
Martinezら[121]は、2Dの関節位置に基づいて3Dの関節位置を回帰させるために、単純だが効果的な完全連結残差ネットワークを提案した。     
当時の最先端の結果を達成したにもかかわらず、この手法は2Dポーズ検出器に過度に依存した再構成の曖昧さのために失敗する可能性があった[118]。    
Tekinら[122]およびZhouら[123]は、3Dポーズを推定するための中間表現として、2Dポーズの代わりに2Dヒートマップを利用した。    
Moreno-Noguer [124]は、2Dと3Dの体のジョイントの距離を2つのユークリッド距離行列（EDM）にエンコードし、距離行列回帰によって人間の3Dポーズを推定した。    
EDMは，画像の平面内での回転や並進に影響されず，また，正規化操作を行った際のスケーリングにも影響されない．    
Wangら[125]は，Pairwise Ranking Convolutional Neural Network（PRCNN）を開発し，一対の人間の関節の深度ランキングを予測した．    
そして、粗い-細かいポーズ推定器を使用して、2Dジョイントと深さランキング行列から3Dポーズを回帰させた。    
Jahangiri and Yuille [126]、Sharma et al.[127]、Li and Lee [128]は、まず、複数の多様な3Dポーズの仮説を生成し、ランキングネットワークを適用して、最適な3Dポーズを選択した。

Given that a human pose can be represented as a graph where the joints are the nodes and the bones are the edges, Graph Convolutional Networks (GCNs) have been applied to the 2D-to-3D pose lifting problem by showing promising performance [129] [130 [131] [132] [133].     
Choi et al. [131] proposed Pose2Mesh, which is a GCN-based method to refine the intermediate 3D pose from its PoseNet.      
With GCN, the MeshNet regresses the 3D coordinates of mesh vertices with graphs constructed from the mesh topology.     
Ci et al. [129] proposed a generic framework, named Locally Connected Network (LCN), which leverages both fully connected network and GCN to encode the relationship between local joint neighborhoods.      
LCN can overcome the limitations of GCN that weight sharing scheme harms pose estimation model’s representation ability, and the structure matrix lacks flexibility to support customized node dependence.      
Zhao et al. [130] also tackled the limitation of the shared weight matrix of convolution filters for all the nodes in GCN.    
A Semantic-GCN was proposed to investigate the  semantic information and relationship.     
The semantic graph convolution (SemGConv) operation is used to learn channelwise weights for edges.    
Both local and global relati onships among nodes are captured since SemGConv and non-local layers are interleaved.      

人間のポーズは，関節をノード，骨をエッジとするグラフとして表現できることから，グラフ畳み込みネットワーク（Graph Convolutional Networks）は，2Dから3Dへのポーズリフティング問題に適用され，有望な性能を示している[129] [130] [131] [132] [133]．    
Choiら[131]は，PoseNetから中間の3Dポーズを絞り込むGCNベースの手法であるPose2Meshを提案した．     
GCNでは、MeshNetがメッシュの頂点の3次元座標を、メッシュのトポロジーから構築されたグラフで回帰させる。    
Ciら[129]は、完全連結ネットワークとGCNの両方を活用して、局所的な結合近傍の関係を符号化するLocally Connected Network (LCN)と呼ばれる汎用フレームワークを提案した。     
これは、完全連結ネットワークとGCNの両方を利用して、局所的な結合近傍の関係を符号化するものである。LCNは、重み共有スキームがポーズ推定モデルの表現力を損ない、構造行列がカスタマイズされたノード依存性をサポートする柔軟性に欠けるというGCNの限界を克服できる。     
Zhao et al. [130]も，GCNでは，畳み込みフィルタの重み行列をすべてのノードで共有するという制限に取り組んでいる．    
意味的な情報や関係を調査するために、Semantic-GCNが提案された。    
セマンティック・グラフ・コンボルーション(SemGConv)演算を用いて，エッジのチャネルごとの重みを学習する．   
SemGConvとノンローカル層がインターリーブされているため、ノード間のローカルおよびグローバルな関係が捕捉される。     
