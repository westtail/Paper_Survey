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
3D HPE datasets are usually collected from controlled environments with selected daily motions.     
It is difficult to obtain the 3D pose annotations for in-the-wild data.     
Thus 3D HPE for in-the-wild data with unusual poses and occlusions is still a challenge.     
To this end, a group of 2D to 3D lifting methods pay attention to estimate the 3D human pose from in-the-wild images without 3D pose annotations such as [109] [134] [135] [110] [111].        
Zhou et al. [109] proposed a weakly supervised transfer learning method that uses 2D annotations of in-the-wild images as weak labels. 3D pose estimation module was connected with intermediate layers of the 2D pose estimation module.      
For in-the-wild images, 2D pose estimation module performed a supervised 2D heatmap regression and a 3D bone length constraint induced loss was applied in the weakly supervised 3D pose estimation module.     
Habibie et al. [134] tailored a projection loss to refine the 3D human pose without 3D annotation.      
A 3D-2D projection module was designed to estimate the 2D body joint locations with the predicted 3D pose from earlier network layer.     
The projection loss was used to update the 3D human pose without requiring 3D annotations.     
nspired by [136], Chen et al. [135] proposed an unsupervised lifting network based on the closure and invariance lifting properties with a geometric self-consistency loss for the lift-reproject-lift process.    
Closure means for a lifted 3D skeleton, after random rotation and re-projection, the resulting 2D skeleton will lie within the distribution of valid 2D pose. Invariance means when changing the viewpoint of 2D projection from a 3D skeleton, the re-lifted 3D skeleton should be the same.     
Instead of estimating 3D human pose from monocular images, videos can provide temporal information to improve accuracy and robustness of 3D HPE, e.g., [137] [138][139] [140] [141] [142] [143] [144].     
Hossain and Little [145] proposed a recurrent neural network using a Long Short-Term Memory (LSTM) unit with shortcut connections to exploit temporal information from sequences of human pose.      
Their method exploits the past events in a sequence-tosequence network to predict temporally consistent 3D pose.     
Noticing that the complementary property between spatial constraints and temporal correlations is usually ignored by prior work, Dabral et al. [139], Cai et al. [142], and Li et al. [146] exploited the spatial-temporal relationships and constraints (e.g., bone-length constraint and left-right symmetry constraint) to improve 3D HPE performance from sequential frames.     
Pavllo et al. [140] proposed a temporal convolution network to estimate 3D pose over 2D keypoints from consecutive 2D sequences.      
However, their method is based on the assumption that prediction errors are temporally non-continuous and independent, which may not hold in presence of occlusions [141].      
Based on [140], Chen et al. [147] added bone direction module and bone length module to ensure human anatomy temporal consistency across video frames, while Liu et al. [148] utilized the attention mechanism to recognize significant frames and model longrange dependencies in large temporal receptive fields.      
Zeng et al. [133] employed the split-and-recombine strategy to address the rare and unseen pose problem.      
The human body is first split into local regions for processing through separate temporal convolutional network branches, then the lowdimensional global context obtained from each branch is combined to maintain global coherence.     

人間のポーズは，関節をノード，骨をエッジとするグラフとして表現できることから，グラフ畳み込みネットワーク（Graph Convolutional Networks）は，2Dから3Dへのポーズリフティング問題に適用され，有望な性能を示している[129] [130] [131] [132] [133]．    
Choiら[131]は，PoseNetから中間の3Dポーズを絞り込むGCNベースの手法であるPose2Meshを提案した．     
GCNでは、MeshNetがメッシュの頂点の3次元座標を、メッシュのトポロジーから構築されたグラフで回帰させる。    
Ciら[129]は、完全連結ネットワークとGCNの両方を活用して、局所的な結合近傍の関係を符号化するLocally Connected Network (LCN)と呼ばれる汎用フレームワークを提案した。     
これは、完全連結ネットワークとGCNの両方を利用して、局所的な結合近傍の関係を符号化するものである。LCNは、重み共有スキームがポーズ推定モデルの表現力を損ない、構造行列がカスタマイズされたノード依存性をサポートする柔軟性に欠けるというGCNの限界を克服できる。     
Zhao et al. [130]も，GCNでは，畳み込みフィルタの重み行列をすべてのノードで共有するという制限に取り組んでいる．    
意味的な情報や関係を調査するために、Semantic-GCNが提案された。    
セマンティック・グラフ・コンボルーション(SemGConv)演算を用いて，エッジのチャネルごとの重みを学習する．   
SemGConvとノンローカル層がインターリーブされているため、ノード間のローカルおよびグローバルな関係が捕捉される。    
3D HPEデータセットは、通常、管理された環境で、選択された日常的な動作を伴って収集されます。    
野生のデータの3Dポーズアノテーションを取得することは困難です。    
そのため、通常とは異なるポーズやオクルージョンを持つ自然界のデータに対する3D HPEは、まだ課題となっている。    
この目的のために、2Dから3Dへのリフティング手法のグループは、3Dポーズアノテーションのない野性の画像から人間の3Dポーズを推定することに注目している[109] [134] [135] [110] [111]。        
Zhouら[109]は、野生の画像の2Dアノテーションを弱いラベルとして利用する弱教師付き伝達学習法を提案している。3Dポーズ推定モジュールは、2Dポーズ推定モジュールの中間層に接続されている。      
野生の画像に対して、2Dポーズ推定モジュールは、教師ありの2Dヒートマップ回帰を行い、弱い教師ありの3Dポーズ推定モジュールでは、3D骨の長さの制約による損失を適用した。    
Habibieら[134]は、3Dアノテーションがなくても、人間の3Dポーズを洗練させるために投影損失を調整した。     
3D-2D 投影モジュールは、以前のネットワーク層から予測された 3D ポーズを用いて、2D の身体関節位置を推定するように設計された。    
投影損失は、3Dアノテーションを必要とせずに人間の3Dポーズを更新するために使用された。    
Chenら[135]は、[136]にヒントを得て、closureとinvarianceのリフティング特性に基づいて、lift-reproject-liftプロセスのための幾何学的自己整合性損失を用いた、教師なしリフティングネットワークを提案した。   
閉鎖性とは、リフティングされた3Dスケルトンに対して、ランダムな回転と再投影を行った後、結果として得られる2Dスケルトンが有効な2Dポーズの分布内に収まることを意味する。不変性（Invariance）とは、3Dスケルトンからの2D投影の視点を変えても、再リフトされた3Dスケルトンは同じであることを意味する。   
単眼画像から人間の3Dポーズを推定する代わりに、動画から時間的な情報を得ることで、3D HPEの精度とロバスト性を向上させることができる。    
HossainとLittle[145]は，人間のポーズのシーケンスから時間情報を利用するために，ショートカット接続を持つLSTM（Long Short-Term Memory）ユニットを用いたリカレント・ニューラル・ネットワークを提案した．     
彼らの手法は、シーケンス-シーケンスネットワークの過去のイベントを利用して、時間的に一貫した3Dポーズを予測する。    
Dabralら[139]、Caiら[142]、Liら[146]は、空間的・時間的な関係と制約（骨の長さの制約や左右の対称性の制約など）を利用して、連続したフレームからの3D HPEの性能を向上させている。    
Pavlloら[140]は，連続する2Dシーケンスから2Dキーポイント上の3Dポーズを推定するために，時間的な畳み込みネットワークを提案した．     
しかし、彼らの方法は、予測誤差が時間的に非連続で独立しているという仮定に基づいており、これはオクルージョンの存在下では成立しない可能性がある[141]。     
Chenら[147]は、[140]に基づいて、ビデオフレーム間の人体解剖学的な時間的一貫性を確保するために、骨の方向モジュールと骨の長さモジュールを追加し、Liuら[148]は、有意なフレームを認識し、大きな時間的受容野における長距離依存性をモデル化するために、注意メカニズムを利用しています。     
Zengら[133]は、希少で見たことのないポーズの問題を解決するために、分割・再結合戦略を採用した。     
これは、人体を局所的な領域に分割して、時間的に別の畳み込みネットワークブランチで処理し、その後、各ブランチから得られた低次元のグローバルコンテキストを結合して、グローバルなコヒーレンスを維持するというものである。    
### Model-based methods. モデルベースの方法。
Model-based methods incorporate parametric body models as noted in Section 2 (such as kinematic model and volumetric model) to estimate human pose and shape as shown in Fig. 5(c).      
The kinematic model is an articulated body representation by connected bones and joints with kinematic constraints, which has gained increasing attention in 3D HPE in recent years.      
Many methods leverage prior knowledge based on the kinematic model such as skeletal joints connectivity information, joints rotation properties, and fixed bone-length ratios for plausible pose estimation, e.g., [149] [19] [150] [151] [152] [153] [154] [155].      
Zhou et al. [149] embedded a kinematic model into a network as kinematic layers to enforce the orientation and rotation constrains.      
Nie et al. [150] and Lee et al. [156] employed a skeleton-LSTM network to leverage joint relations and connectivity.     
Observing that human body parts have a distinct degree of freedom (DOF) based on the kinematic structure, Wang et al. [151] and Nie et al. [154] proposed bidirectional networks to model the kinematic and geometric dependencies of the human skeleton.        
Kundu et al. [152] [157] designed a kinematic structure preservation approach by inferring local-kinematic parameters with energy-based loss and explored 2D part segments based on the parent-relative local limb kinematic model.      
Xu et al. [153] demonstrated that noisy 2D joint is one of the key obstacles for accurate 3D pose estimation.     
Hence a 2D pose correction module was employed to refine unreliable 2D joints based on the kinematic structure.      
Zanfir et al. [158] introduced a kinematic latent normalizing flow representation (a sequence of invertible transformations applied to the original distribution) with differentiable semantic body part alignment loss functions.   
Compared with the kinematic model, which produces human poses or skeletons, volumetric models can recover high-quality human mesh, providing extra shape information of human body.       
As one of the most popular volumetric models, the SMPL model [25] has been widely used in  3D HPE, e.g., [159] [160] [161] [162] [163] [164] [165] [166] [167] [168], because it is compatible with existing rendering engines.       
Tan et al. [161], Tung et al. [162], Pavlakos et al. [169], and Omran et al. [170] regressed SMPL parameters to reconstruct 3D human mesh.      
Instead of predicting SMPL parameters, Kolotouros et al. [171] regressed the locations of the SMPL mesh vertices using a Graph-CNN architecture.       
Zhu et al. [172] combined the SMPL model with a hierarchical mesh deformation framework to enhance the flexibility of free-form 3D deformation.      
Kundu et al. [173] included a colorrecovery module in the SMPL model to obtain vertex color via reflectional symmetry.     
Arnab et al. [113] pointed out that methods using the SMPL model usually fail on the in-thewild data.       
They employed the bundle adjustment method to cope with occlusion, unusual poses and object blur.     
Doersch and Zisserman [165] proposed a transfer learning method to regress SMPL parameters by training on the synthetic human video dataset SURREAL [174].      
Kocabas et al. [175] included the large-scale motion capture dataset AMASS [176] for adversarial training of their SMPL-based method named VIBE (Video Inference for Body Pose and Shape Estimation).      
VIBE leveraged AMASS to discriminate between real human motions and predicted pose by pose regression module.      
Since low-resolution visual content is more common in real-world scenarios than the high-resolution visual content, existing well-trained models may fail when resolution is degraded.      
Xu et al. [177] introduced the contrastive learning scheme into self-supervised resolution-aware SMPL-based network.     
The self-supervised contrastive learning scheme uses a selfsupervision loss and a contrastive feature loss to enforce the feature and scale consistency.      
There are several extended SMPL-based models to address the limitations of the SMPL model such as high computational complexity, and lack of hands and facial landmarks.        
Bogo et al. [159] proposed SMPLify to estimate 3D human mesh, which fits the SMPL model to the detected 2D joints and minimizes the re-projection error.       
An extended version of SMPLify was presented by Lassner et al. [160].   
The running time is reduced by employing a random forest regression to regress SMPL parameters, but it still cannot achieve real-time throughput.      
Kanazawa et al. [178] further proposed an adversarial learning approach to directly infer SMPL parameters in real-time. Pavlakos et al. [179] introduced a new model, named SMPL-X, that can also predict fully articulated hands and facial landmarks.      
Following the SMPLify method, they also proposed SMPLifyX, which is an improved version learned from AMASS dataset [176].       
Hassan et al. [163] further extended SMPLify-X to PROX – a method enforcing Proximal Relationships with Object eXclusion by adding 3D environmental constraints.      
Kolotouros et al. [164] integrated the regression-based and optimization-based SMPL parameter estimation methods to a new one named SPIN (SMPL oPtimization IN the loop) while employing SMPLify in the training loop.     
Osman et al. [180] upgraded SMPL to STAR by training with additional 10,000 scans for better model generalization.     
The number of model parameters is reduced to 20% of that of SMPL.     
Instead of using the SMPL-based model, other volumetric models have also been used for recovering 3D human mesh, e.g., [181] [182] [183] [184].       
Chen et al. [182] introduced a Cylinder Man Model to generate occlusion labels for 3D data and performed data augmentation.     
A pose regularization term was introduced to penalize wrong estimated occlusion labels.        
Xiang et al. [183] utilized the Adam model [30] to reconstruct the 3D motions. A 3D human representation, named 3D Part Orientation Fields (POFs), was introduced to encode the 3D orientation of human body parts in the 2D space.     
Wang et al. [185] presented a new Bone-level Skinned Model of human mesh, which decouples bone modelling and identity-specific variations by setting bone lengths and joint angles.      
Fisch and Clark [186] introduced an orientation keypoints model which can compute full 3-axis joint rotations including yaw, pitch, and roll for 6D HPE.

モデルベースの手法では、図5（c）に示すように、第2章で述べたパラメトリックなボディモデル（キネマティックモデルやボリュームモデルなど）を用いて、人間の姿勢や形状を推定する。     
キネマティックモデルとは、骨や関節を運動学的に拘束して連結した多関節体の表現であり、近年の3D HPEでは注目されている。     
多くの手法は，骨格の関節接続情報，関節の回転特性，固定された骨の長さの比率など，運動モデルに基づく事前知識を活用して，妥当なポーズ推定を行っているが，例えば，[149] [19] [150] [151] [152] [153] [154] [155]などがある．     
Zhouら[149]は、キネマティック・モデルをキネマティック・レイヤーとしてネットワークに組み込み、姿勢と回転の制約を強化している。     
Nieら[150]とLeeら[156]は，関節の関係と接続性を利用するために，スケルトン-LSTMネットワークを採用した．    
人体の各部位が運動構造に基づいて明確な自由度（DOF）を持っていることに着目し、Wangら[151]およびNieら[154]は、人体骨格の運動依存性および幾何学的依存性をモデル化する双方向ネットワークを提案した。       
Kundu ら [152] [157] は、エネルギーベースのロスを用いて局所運動パラメータを推論することで、運動構造保存アプローチを設計し、親-相対的な局所四肢の運動モデルに基づいて 2D パーツセグメントを探索した。     
Xuら[153]は、ノイズの多い2D関節が正確な3Dポーズ推定のための重要な障害の1つであることを示した。    
そこで、2Dポーズ補正モジュールを用いて、信頼性のない2Dジョイントを運動学的構造に基づいて改良した。     
Zanfirら[158]は、微分可能なセマンティック身体部位アライメント損失関数を用いて、運動学的な潜在的正規化フロー表現（元の分布に適用される反転可能な変換のシーケンス）を導入した。   
人間のポーズや骨格を生成するキネマティックモデルと比較して，ボリューメトリックモデルは高品質の人間のメッシュを復元することができ，人体の余分な形状情報を提供することができる．      
最も人気のあるボリュームモデルの1つであるSMPLモデル[25]は，既存のレンダリングエンジンとの互換性があるため，例えば[159] [160] [161] [162] [163] [164] [165] [166] [167] [168]のように，3D HPEで広く使用されています．      
Tanら[161]，Tungら[162]，Pavlakosら[169]，Omranら[170]は，SMPLパラメータを回帰させて，人間の3次元メッシュを再構成しています．     
SMPLのパラメータを予測する代わりに Kolotourosら[171]は，SMPLパラメータを予測する代わりに，SMPLメッシュの頂点の位置を回帰しました．Graph-CNNアーキテクチャを用いて，SMPLメッシュの頂点の位置を回帰しました．     
Zhuら[172]は，SMPLモデルと階層的なメッシュ変形フレームワークを組み合わせることで，自由形状による3次元変形の柔軟性を高めている．     
Kunduら[173]は，SMPLモデルに色回復モジュールを組み込み，反射対称性によって頂点の色を得るようにした．    
Arnabら[113]は，SMPLモデルを用いた手法は，イン・ザ・ワイルド・データでは失敗することが多いと指摘している．      
Arnabら[113]は，SMPLモデルを用いた手法は，in-the-wildデータでは失敗することが多いと指摘し，オクルージョン，異常なポーズ，物体のブレに対処するためにバンドル調整法を採用した．    
Doersch and Zisserman [165]は，人間の合成映像データセット SURREAL [174]で学習することで，SMPLパラメータを回帰させる伝達学習法を提案した．     
Kocabasら[175]は，VIBE（Video Inference for Body Pose and Shape Estimation）と名付けたSMPLベースの手法の敵対的学習のために，大規模モーションキャプチャデータセットAMASS[176]を用いた．     
VIBEは、AMASSを利用して、実際の人間の動きと、ポーズ回帰モジュールによって予測されたポーズとを識別する。     
実世界のシナリオでは、高解像度のビジュアル・コンテンツよりも低解像度のビジュアル・コンテンツの方が一般的であるため、解像度が低下すると既存の十分に学習されたモデルが失敗する可能性がある。     
Xuら[177]は，自己教師付き解像度認識SMPLベースのネットワークにコントラスト学習スキームを導入した．    
自己教師付きコントラスト学習スキームは、自己教師損失とコントラスト特徴損失を使用して、特徴とスケールの一貫性を確保します。     
計算量が多い、手や顔のランドマークがないなどのSMPLモデルの限界を解決するために、いくつかの拡張SMPLベースモデルがあります。       
Bogoら[159]は，人間の3次元メッシュを推定するためにSMPLifyを提案しており，検出された2次元関節にSMPLモデルを適合させ，再投影誤差を最小化している．      
Lassnerら[160]は，SMPLifyの拡張版を発表した．     
SMPLパラメータの回帰にランダムフォレスト回帰を採用することで実行時間を短縮していますが，それでもリアルタイムのスループットを実現することはできません．     
Kanazawaら[178]はさらに，SMPLパラメータをリアルタイムで直接推論するための敵対的学習アプローチを提案した。Pavlakosら[179]は，SMPL-Xと名付けた新しいモデルを導入し，完全に関節のある手や顔のランドマークも予測できるようにした．     
彼らは，SMPLify 法に続き，AMASS データセット[176]から学習した改良版である SMPLifyX も提案している．      
Hassanら[163]は、SMPLify-Xをさらに拡張し、3D環境制約を加えることでProximal Relationships with Object eXclusionを強化する手法であるPROXを提案している。     
Kolotourosら[164]は，回帰ベースと最適化ベースのSMPLパラメータ推定法をSPIN（SMPL oPtimization IN the loop）という新しい手法に統合し，SMPLifyを学習ループに採用した．    
Osman et al. [180]は，SMPLをSTARにアップグレードしました．10,000スキャンを追加して学習することで、モデルの汎用性を高めました。    
モデルパラメータの数は モデルのパラメータ数は、SMPLの20%に減少しました。    
SMPLに基づいたモデルを使用する代わりに，他のボリューメトリック SMPLに基づくモデルの代わりに，他の体積モデルも3D人体メッシュの復元に使用されています．例えば，[181] [182] [183] [184]などである．      
Chenら[182]は，オクルージョンを生成するために Cylinder Man Modelを導入し，3Dデータのオクルージョンラベルを生成して を導入し，データの補強を行った．    
ポーズ正則化 誤って推定されたオクルージョンラベルにペナルティを課すために、ポーズ正則化項が導入された。ラベルを生成する。       
Xiangら[183]は、Adamモデル[30]を利用して、3Dモーションを再構成した。また、3D Part Orientation Fields（POF）と呼ばれる3D人体表現を導入し、2D空間における人体パーツの3Dオリエンテーションを符号化した。   
Wangら[185]は、骨の長さと関節の角度を設定することで、骨のモデリングとアイデンティティ特有のバリエーションを切り離す、新しいBone-level Skinned Model of human meshを発表した。     
Fisch and Clark [186]は、6D HPE のためにヨー、ピッチ、ロールを含む完全な 3 軸関節回転を計算できるオリエンテーション・キーポイント・モデルを導入した。

 
### Multi-person 3D HPE 複数人での3D HPE
Fig. 6: Illustration of the multi-person 3D HPE frameworks.     
(a) Top-Down approaches first detect single-person regions by human detection network.     
For each single-person region, individual 3D pose can be estimated by 3D pose network.      
Then all 3D poses are aligned to the world coordinate.      
(b) Bottom-Up approaches first estimate all body joints and depth maps, then associate body parts to each person according to the root depth and part relative depth.    
Part of the figure is from [187].     

図6：複数人対応の3D HPEフレームワークの説明。    
(a) トップダウン型のアプローチでは、まず人物検出ネットワークによって一人の人物の領域を検出します。    
各一人の人物領域に対して、3Dポーズネットワークにより個々の3Dポーズが推定されます。     
そして、すべての3Dポーズをワールド座標に合わせます。     
(b) ボトムアップアプローチでは、まず、すべてのボディジョイントとデプスマップを推定し、次に、ルートデプスとパーツ相対デプスに従って、ボディパーツを各人に関連付ける。   
図の一部は[187]からの引用である。  

For 3D multi-person HPE from monocular RGB images or videos, similar categories as 2D multi-person HPE are noted here: top-down approaches and bottom-up approaches as shown in Fig. 6 (a) and Fig. 6 (b), respectively. 
The comparison between 2D top-down and bottom-up approaches in Section 3.2 is also applicable for the 3D case.
単眼のRGB画像や動画からの3D多人数HPEについても、2D多人数HPEと同様に、図6(a)に示すようにトップダウン型、図6(b)に示すようにボトムアップ型のアプローチがあります。
なお、3.2節のトップダウン方式とボトムアップ方式の比較は、3Dの場合にも当てはまります。

#### Top-down approaches.    トップダウン方式。   
Top-down approaches of 3D multi-person HPE first perform human detection to detect each individual person.      
Then for each detected person, absolute root (center joint of the human) coordinate and 3D root-relative pose are estimated by 3D pose networks.      
Based on the absolute root coordinate of each person and their rootrelative pose, all poses are aligned to the world coordinate.     
Rogez et al. [188] localized candidate regions of each person to generate potential poses, and used a regressor to jointly refine the pose proposals.      
This localization-classificationregression method, named LCR-Net, performed well on the controlled environment datasets but could not generalize well to in-the-wild images.     
To address this issue, Rogez et al. [189] proposed LCR-Net++ by using synthetic data augmentation for the training data to improve performance.      
Zanfir et al. [190] added semantic segmentation to the 3D multiperson HPE module with scene constraints.      
Additionally, the 3D temporal assignment problem was tackled by the Hungarian matching method for video-based multi-person 3D HPE.       
Moon et al. [191] introduced a camera distanceaware approach that the cropped human images were fed into their developed RootNet to estimate the camera-centered coordinates of human body’s roots.      
Then the root-relative 3D pose of each cropped human was estimated by the proposed PoseNet.       
Benzine et al. [192] proposed a single-shot approach named PandaNet (Pose estimAtioN and Detection Anchorbased Network).      
A low-resolution anchor-based representation was introduced to avoid the occlusion problem.      
A poseaware anchor selection module was developed to address the overlapping problem by removing the ambiguous anchors.      
An automatic weighting of losses associated with different scales was used to handle the imbalance issue of different sizes of people.      
Li et al. [193] tackled the lack of global information in the top-down approaches.     
They adopted a Hierarchical Multi-person Ordinal Relations method to leverage body level semantic and global consistency for encoding the interaction information hierarchically.

3DマルチパーソンHPEのトップダウンアプローチでは、まず人物検出を行い、個々の人物を検出します。     
次に、検出された各人物について、3Dポーズネットワークにより、絶対根元（人物の中心関節）座標と3D根元-相対ポーズが推定されます。     
各人物の絶対根元座標と根元相対ポーズに基づいて、すべてのポーズをワールド座標に整列させる。    
Rogezら[188]は，各人物の候補領域をローカライズしてポーズの候補を生成し，回帰器を用いてポーズ案を共同で精査した．     
LCR-Net と名付けられたこの局在化-分類-回帰法は、制御された環境のデータセットでは良好な性能を示したが、野生の画像にはうまく一般化できなかった。    
この問題を解決するために、Rogezら[189]は、性能を向上させるために、学習データに合成データの補強を用いるLCR-Net++を提案した。     
Zanfirら[190]は、シーン制約のある3D多人数HPEモジュールにセマンティックセグメンテーションを追加した。     
さらに、ビデオベースの多人数3D HPEのためのハンガリアンマッチング法により、3Dの時間的割り当て問題に取り組んだ。      
Moonら[191]は、カメラ距離を考慮したアプローチを導入しています。これは、切り取られた人間の画像を、彼らが開発したRootNetに供給し、人体の根元のカメラ中心座標を推定するというものです。     
そして、提案されたPoseNetによって、切り取られた人間の根元に関連した3Dポーズが推定されました。      
Benzineら[192]は、PandaNet（Pose estimAtioN and Detection Anchorbased Network）と名付けたシングルショットのアプローチを提案した。     
オクルージョンの問題を回避するために，低解像度のアンカーベースの表現を導入した．     
曖昧なアンカーを除去することで重複問題を解決するために、ポーズを考慮したアンカー選択モジュールを開発しました。     
また，異なるスケールに関連する損失を自動的に重み付けすることで，異なるサイズの人の不均衡問題を処理しました．     
Li et al. [193] は，トップダウンアプローチにおけるグローバルな情報の欠如に取り組んだ．    
彼らは、階層的多人数順序関係法を採用し、身体レベルのセマンティックとグローバルな一貫性を活用して、インタラクション情報を階層的に符号化した。

#### Bottom-up approaches.    ボトムアップ・アプローチ
In contrast to the top-down approaches, bottom-up approaches first produce all body joint locations and depth maps, then associate body parts to each person according to the root depth and part relative depth.       
A key challenge of bottom-up approaches is how to group human body joints belonging to each person.      
Zanfir et al. [194] formulated the person grouping problem as a binary integer programming (BIP) problem.      
A limb scoring module was used to estimate candidate kinematic connections of detected joints and a skeleton grouping module assembled limbs into skeletons by solving the BIP problem.       
Nie et al. [101] proposed a Single-stage multi-person Pose Machine (SPM) to define the unique identity root joint for each person.    
The body joints were aligned to each root joint by using the dense displacement maps.       
However, this method is limited in that only paired 2D images and 3D pose annotations could be used for supervised learning.      
Without paired 2D images and 3D pose annotations, Kundu et al. [195] proposed a frozen network to exploit the shared latent space between two diverse modalities under a practical deployment paradigm such that the learning could be cast as a cross-model alignment problem.    
Fabbri et al. [196] developed a distancebased heuristic for linking joints in a multi-person setting.       
Specifically, starting from detected heads (i.e., the joint with the highest confidence), the remaining joints are connected by selecting the closest ones in terms of 3D Euclidean distance.     
Another challenge of bottom-up approaches is occlusion.      
To cope with this challenge, Metha et al. [197] developed an Occlusion-Robust Pose-Maps (ORPM) approach to incorporate redundancy into the location-maps formulation, which facilitates person association in the heatmaps especially for occluded scenes.     
Zhen et al. [187] leveraged a depth-aware part association algorithm to assign joints to individuals by reasoning about inter-person occlusion and bone-length constraints.      
Mehta et al. [198] quickly inferred intermediate 3D pose of visible body joints regardless of the accuracy.     
Then the completed 3D pose is reconstructed by inferring occluded joints using learned pose priors and global context.     
The final 3D pose was refined by applying temporal coherence and fitting the kinematic skeletal model.    
トップダウンアプローチとは対照的に、ボトムアップアプローチでは、まず、すべての身体関節の位置と深度マップを作成し、次に、根元の深度とパーツの相対的な深度に応じて、各人に身体パーツを関連付けます。      
ボトムアップ・アプローチの重要な課題は、各人に属する人体の関節をどのようにグループ化するかである。     
Zanfirら[194]は、人物グルーピング問題を二元整数計画（BIP）問題として定式化しました。     
肢体スコアリングモジュールは、検出された関節の運動学的接続の候補を推定するために使用され、スケルトングルーピングモジュールは、BIP問題を解くことによって肢体をスケルトンに組み立てる。      
Nie et al. [101]は、各人に固有のアイデンティティを持つルートジョイントを定義するために、Single-stage multi-person Pose Machine (SPM)を提案した。   
体の関節は，密な変位マップを用いて，各根元関節に整列させた．      
しかし、この方法では、ペアの2D画像と3Dポーズのアノテーションのみが教師付き学習に使用できるという制限があります。     
Kunduら[195]は、ペアの2D画像と3Dポーズのアノテーションがなくても、2つの多様なモダリティ間の共有された潜在的な空間を利用するために、実用的な展開パラダイムの下で、モデル間のアライメント問題として学習を行うことができるフローズンネットワークを提案した。   
Fabbriら[196]は、複数の人がいる環境で関節を連結するための距離ベースのヒューリスティックを開発した。      
具体的には、検出された頭部（すなわち、最も信頼性の高い関節）から始まり、残りの関節は、3Dユークリッド距離の観点から最も近いものを選択して接続される。    
ボトムアップアプローチのもう一つの課題は、オクルージョンです。     
この課題に対処するため、Methaら[197]は、Occlusion-Robust Pose-Maps (ORPM)アプローチを開発し、冗長性をロケーション・マップの定式化に組み込むことで、特にオクルージョンのあるシーンでのヒートマップにおける人物の関連付けを容易にしました。    
Zhenら[187]は、奥行きを考慮したパーツ・アソシエーション・アルゴリズムを活用して、人物間のオクルージョンと骨の長さの制約を推論することで、個人にジョイントを割り当てています。     
Mehtaら[198]は、精度に関わらず、可視化されたボディジョイントの中間的な3Dポーズを素早く推定する。    
そして、学習したポーズプライアとグローバルコンテキストを用いて、オクルージョンしたジョイントを推論することで、完成した3Dポーズを再構築する。    
最終的な3Dポーズは、時間的なコヒーレンスを適用し、運動学的な骨格モデルをフィットさせることで洗練された。   

#### Comparison of top-down and bottom-up approaches. トップダウン・アプローチとボトムアップ・アプローチの比較
Top-down approaches usually achieve promising results by relying on the state-of-the-art person detection methods and single-person pose estimation methods.     
But the computational complexity and the inference time may become excessive with the increase in the number of humans, especially in crowded scenes.      
Moreover, since top-down approaches first detect the bounding box for each person, global information in the scene may get neglected.      
The estimated depth of cropped region may be inconsistent with the actual depth ordering and the predicted human bodies may be placed in overlapping positions. On the contrary, the bottom-up approaches enjoy linear computation and time complexity.      
However, if the goal is to recover 3D body mesh, it is not straightforward for the bottom-up approaches to reconstruct human body meshes.       
For top-down approaches, after detecting each individual person, human body mesh of each person can be easily recovered by incorporating the model-based 3D single-person HPE estimator.      
While for the bottom-up approaches, additional model regressor module is needed to reconstruct human body meshes based on the final 3D poses.       

トップダウン・アプローチは、通常、最先端の人物検出手法や一人の人物のポーズ推定手法に頼ることで、有望な結果を得ることができます。          
しかし、特に混雑したシーンでは、人間の数が増えるにつれて、計算量や推論時間が過大になる可能性があります。     
また、トップダウン方式では、まず各人物のバウンディングボックスを検出するため、シーンのグローバルな情報が無視されてしまう可能性があります。     
また，切り取られた領域の奥行きの推定値が実際の奥行きの順序と一致しなかったり，予測された人体が重複して配置されたりすることがあります．一方、ボトムアップ方式では、計算量と時間が直線的になります。     
しかし、3Dボディメッシュの復元を目的とした場合、ボトムアップアプローチで人体メッシュを復元することは容易ではありません。      
トップダウン型のアプローチでは、個々の人物を検出した後、モデルベースの3D一人用HPE推定器を組み込むことで、各人物の人体メッシュを容易に復元することができます。     
一方、ボトムアップ型のアプローチでは、最終的な3Dポーズに基づいて人体メッシュを再構成するために、追加のモデル回帰モジュールが必要となります。     

### 4.1.2 Multi-view 3D HPE マルチビュー3D HPE
The partial occlusion is a challenging problem for 3D HPE in the single-view setting.     
The natural solution to overcome this problem is to estimate 3D human pose from multiple views, since the occluded part in one view may become visible in other views.     
In order to reconstruct the 3D pose from multiple views, the association of corresponding location between different cameras needs to be resolved.     
A group of methods [199] [200] [201] [202] [203] used body models to tackle the association problem by optimizing model parameters to match the model projection with the 2D pose.       
The widely used 3D pictorial structure model [204] is such a model.      
However, these methods usually need large memory and expensive computational cost, especially for multi-person 3D HPE under multi-view settings.        
Rhodin et al. [205] employed a multi-view consistency constraint in the network, however it requires a large amount of 3D groundtruth training data.       
To overcome this limitation, Rhodin et al. [206] further proposed an encoder-decoder framework to learn the geometry-aware 3D latent representation from multi-view images and background segmentation without 3D annotations.      
Chen et al. [207], Dong et al. [202], Chen et al. [208], Mitra et al. [209], Iqbal et al. [210], Zhang et al. [211], and Huang et al. [212] proposed multi-view matching frameworks to reconstruct 3D human pose across all viewpoints with consistency constraints.      
Pavlakos et al. [199] and Zhang et al. [213] aggregated the 2D keypoint heatmaps of multi-view images into a 3D pictorial structure model based on all the calibrated camera parameters.      
However, when multi-view camera environments change, the model needs to be retrained.         
Liang et al. [201] and Habermann et al. [214] inferred the non-rigid 3D deformation parameters to reconstruct a 3D human body mesh from multi-view images.       
Kadkhodamohammadi and Padoy [215], Qiu et al. [200], and Kocabas et al. [216] employed epipolar geometry to match paired multi-view poses for 3D pose reconstruction and generalized their methods to new multi-view camera environments.      
It should be noted that matching each pair of views separately without the cycle consistency constraint may lead to incorrect 3D pose reconstructions [202].     
Tu et al. [203] aggregated all the features in each camera view in the 3D voxel space to avoid incorrect estimation in each camera view.      
A cuboid proposal network and a pose regression network were designed to localize all people and to estimate the 3D pose, respectively.      
When given sufficient viewpoints (more than ten), it is not practical to use all viewpoints for 3D pose estimation.       
Pirinen et al. [217] proposed a selfsupervised reinforcement learning approach to select a small set of viewpoints to reconstruct the 3D pose via triangulation.     
Besides accuracy, the lightweight architecture, fast inference time, and efficient adaptation to new camera settings also need to be taken into consideration in multi-view HPE.      
In contrast to [202] which matched all view inputs together, Chen et al. [218] applied an iterative processing strategy to match 2D poses of each view with the 3D pose while the 3D pose was updated iteratively.      
Compared to the previous methods whose running time may explode with the increase in the number of cameras, their method has linear time complexity.       
Remelli et al. [219] encoded images of each view into a unified latent representation so that the feature maps were disentangled from camera viewpoints.     
As a lightweight canonical fusion, these 2D representations are lifted to the 3D pose using a GPU-based Direct Linear Transform to accelerate the processing.     
In order to improve the generalization ability of multi-view fusion scheme, Xie et al. [220] proposed a pre-trained multi-view fusion model (MetaFuse), which can be efficiently adapted to new camera settings with few labeled data.      
They deployed the modelagnostic meta-learning framework to learn the optimal initialization of the generic fusion model for adaptation.

部分的なオクルージョンは、単視点での3D HPEにとって難しい問題です。    
この問題を解決するための自然な方法は、複数のビューから人間の3Dポーズを推定することです。なぜなら、1つのビューで隠蔽された部分が他のビューでは見えることがあるからです。    
複数のビューから3Dポーズを再構成するためには、異なるカメラ間の対応する位置の関連性を解決する必要がある。    
ボディモデルを用いて、モデルのパラメータを最適化し、モデルの投影と2Dポーズを一致させることで、関連付けの問題に取り組む手法がある [199] [200] [201] [202] [203]。      
広く使われている3次元絵画構造モデル[204]は、このようなモデルである。     
しかし、これらの方法は、特にマルチビュー設定の下での複数人の3D HPEのために、通常、大容量のメモリと高価な計算コストを必要とする。       
Rhodinら[205]は、ネットワークに多視点一貫性制約を採用しているが、大量の3Dグランドトゥルースの学習データが必要である。      
この限界を克服するために、Rhodinら[206]は、マルチビュー画像と3Dアノテーションのない背景セグメンテーションからジオメトリを考慮した3D潜在表現を学習するエンコーダ・デコーダフレームワークをさらに提案した。     
Chenら[207]、Dongら[202]、Chenら[208]、Mitraら[209]、Iqbalら[210]、Zhangら[211]、Huangら[212]は、一貫性制約のあるすべての視点で人間の3Dポーズを再構成するマルチビューマッチングフレームワークを提案した。     
Pavlakosら[199]とZhangら[213]は、マルチビュー画像の2Dキーポイントヒートマップを、較正されたすべてのカメラパラメータに基づいて、3D絵画構造モデルに集約した。     
しかし、マルチビューカメラの環境が変化すると、モデルを再学習する必要がある。        
Liangら[201]とHabermannら[214]は，マルチビュー画像から3D人体メッシュを再構成するために，非剛体の3D変形パラメータを推論している．      
Kadkhodamohammadi and Padoy [215]，Qiu et al. [200]，Kocabas et al. [216]は，エピポーラ幾何学を用いて，3D ポーズ再構成のためにマルチビューのペアポーズをマッチングし，その手法を新しいマルチビューカメラ環境に一般化した．   
なお，周期整合性制約を用いずに各ビューのペアを個別にマッチングすると，誤った3次元ポーズ再構成につながる可能性があることに注意が必要である[202]．     
Tuら[203]は、各カメラ・ビューでの誤った推定を避けるために、各カメラ・ビューのすべての特徴を3Dボクセル空間に集約した。     
人をローカライズするために立方体提案ネットワークを、3Dポーズを推定するためにポーズ回帰ネットワークをそれぞれ設計した。     
十分な視点（10以上）が与えられた場合、3Dポーズ推定のためにすべての視点を使用することは実用的ではない。      
Pirinenら[217]は，自己教師付き強化学習法を提案し，小さな視点セットを選択して Pirinenら[217]は，三角測量によって3Dポーズを再構成するために，小さな視点のセットを選択する自己教師付き強化学習アプローチを提案した．     
マルチビューHPEでは、精度に加えて、軽量なアーキテクチャ、高速な推論時間、新しいカメラ設定への効率的な適応なども考慮する必要がある。     
Chenら[218]は、すべてのビューの入力をまとめて照合する[202]とは対照的に、反復処理戦略を適用して、3Dポーズを反復的に更新しながら、各ビューの2Dポーズを3Dポーズに照合しています。     
カメラの数が増えると実行時間が爆発的に増加する可能性がある従来の手法と比較して、彼らの手法は線形の時間的複雑さを持っています。      
Remelliら[219]は，各ビューの画像を統一的な潜在表現にエンコードし，特徴マップがカメラの視点から切り離されるようにしました．    
軽量な正準融合として、GPUベースのDirect Linear Transformを用いて、これらの2D表現を3Dポーズにリフトアップし、処理を高速化しています。     
Xieら[220]は、多視点融合方式の汎用性を向上させるために、事前に学習された多視点融合モデル（MetaFuse）を提案しました。このモデルは、少ないラベル付きデータで新しいカメラ設定に効率的に適応することができます。   
彼らは、モデル不可知論的なメタ学習フレームワークを展開して、一般的な融合モデルの最適な初期化を学習しました。融合モデルの最適な初期化を学習するために、モデル不可知論的なメタ学習フレームワークを導入しました。

### 4.2 3D HPE from other sources 他のソースからの3D HPE
Although monocular RGB camera is the most common device used for 3D HPE, other sensors (e.g., depth sensor, IMUs, and radio frequency device) are also used for this purpose.    

3D HPEには、単眼のRGBカメラが最も一般的なデバイスですが 3D HPEには単眼RGBカメラが最も一般的ですが、その他のセンサー（深度センサー、IMU, や高周波デバイスなど）も利用されています。   

#### Depth and point cloud sensors:  深度センサーと点群センサー 
Depth sensors have gained more attention recently for conducting 3D computer vision tasks due to their low-cost and increased utilization.      
As one of the key challenges in 3D HPE, the depth ambiguity problem can be alleviated by utilizing depth sensors.     
Yu et al. [221] presented a single-view and real-time method named DoubleFusion to estimate 3D human pose from a single
depth sensor without using images.     
The inner body layer was used to reconstruct 3D shape by volumetric representation, and the outer layer updated the body shape and pose by fusing more geometric details.      
Xiong et al. [222] proposed an Anchor-to-Joint regression network (A2J) using depth images. 3D joints positions were estimated by integrating estimated multiple anchor points with global-local spatial context information.      
Kadkhodamohammadi et al. [223] used multi-view RGB-D cameras to capture color images with depth information in the real operating room environments.       
A random forest-based prior was deployed to incorporate priori environment information.      
The final 3D pose was estimated by multi-view fusion and RGB-D optimization.      
Zhi et al. [224] reconstructed detailed meshes with highresolution albedo texture from RGB-D video.       
Compared with depth images, point clouds can provide more information.     
The state-of-the-art point cloud feature extraction techniques, PointNet [225] and PointNet++ [226], have demonstrated excellent performance for classification and segmentation tasks.     
Jiang et al. [227] combined PointNet++ with the SMPL body model to regress 3D human pose.      
A modified PointNet++ with a graph aggregation module can extract more useful unordered features.     
After mapping into ordered skeleton joint features by an attention module, a skeleton graph module extracts ordered features to regress SMPL parameters for accurate 3D pose estimation.      
Wang et al. [228] presented PointNet++ with a spatial-temporal mesh attention convolution method to predict 3D human meshes with refinement.

深度センサーは、低コストで利用しやすいことから、3Dコンピュータビジョンのタスクを遂行するために近年注目されています。     
3D HPEにおける重要な課題の1つである奥行きの曖昧さの問題は、奥行きセンサを利用することで軽減することができます。    
Yuら[221]は，画像を使わずに1つの深度センサから人間の3次元姿勢を推定する，DoubleFusionと名付けられたシングルビューのリアルタイム手法を発表しました．
Yuら[221]は，画像を使用せずに1つの深度センサから人間の3Dポーズを推定する単視点・リアルタイム手法DoubleFusionを発表した．    
内側のボディレイヤーはボリューム表現により3D形状を再構築し、外側のレイヤーはより多くの幾何学的な詳細を融合することでボディ形状とポーズを更新した。     
Xiongら[222]は，深度画像を用いたAnchor-to-Joint regression network（A2J）を提案した．推定された複数のアンカーポイントを、グローバル・ローカルな空間コンテキスト情報と統合することで、3D関節位置を推定した。     
Kadkhodamohammadiら[223]は，マルチビューRGB-Dカメラを用いて，実際の手術室環境における奥行き情報を含むカラー画像を撮影した．      
また，ランダムフォレストに基づく事前処理により，環境情報を事前に取り込んだ．     
最終的な3Dポーズは、多視点融合とRGB-D最適化によって推定された。     
Zhiら[224]は，RGB-Dビデオから高解像度のアルベドテクスチャを持つ詳細なメッシュを再構成した．      
深度画像と比較して，点群はより多くの情報を提供することができる．    
最先端の点群特徴抽出技術であるPointNet [225]およびPointNet++ [226]は、分類やセグメンテーションのタスクにおいて優れた性能を発揮します。    
Jiangら[227]は、PointNet++とSMPLのボディモデルを組み合わせて、人間の3Dポーズを回帰させています。     
グラフ集約モジュールを用いて改良したPointNet++は、より有用な順序付けられていない特徴を抽出することができる。    
注意モジュールによって順序付けられたスケルトン・ジョイント特徴にマッピングされた後、スケルトン・グラフ・モジュールが順序付けられた特徴を抽出し、正確な3Dポーズ推定のためにSMPLパラメータを回帰する。     
Wangら[228]は、PointNet++と空間的・時間的メッシュ注目コンボリューション法を用いて、洗練された人間の3Dメッシュを予測する方法を紹介しています。

#### IMUs with monocular images: 単眼鏡で撮影されたIMU 
 
Wearable Inertial Measurement Units (IMUs) can track the orientation and acceleration of specific human body parts by recording motions without object occlusions and clothes obstructions.      
However, the drifting problem may occur overtime when using IMUs.      
Marcard et al. [229] proposed the Sparse Inertial Poser (SIP) to reconstruct human pose from 6 IMUs attached to the human body.      
The collected information was fitted into the SMPL body model with coherence constraints to obtain accurate results.     
Marcard et al. [230] further associated 6-17 IMU sensors with a hand-held moving camera for in-the-wild 3D HPE.    
A graph-based optimization method was introduced to assign each 2D person detection to a 3D pose candidate from long-range frames.      
Huang et al. [231] addressed the limitation of the Sparse Inertial Poser (SIP) method [229].     
Multiple pose parameters can generate the same IMU orientation, also collecting IMUs data is time-consuming. 
Thus, a large synthetic dataset was created by placing virtual sensors on the SMPL mesh to obtain orientations and accelerations from motion capture sequences of AMASS dataset [176]. 
A bi-directional RNN framework was proposed to map IMU orientations and accelerations to SMPL parameters with past and future information.       
Zhang et al. [232] introduced an orientation regularized pictorial structure model to estimate 3D pose from multi-view heatmaps associated with IMUs orientation.      
Huang et al. [233] proposed DeepFuse, a twostage approach by fusing IMUs data with multi-view images. The first stage only processes multi-view images to predict a volumetric representation and the second stage uses IMUs to refine the 3D pose by an IMU-bone refinement layer.      
Radio frequency device: Radio frequency (RF) based sensing technology has also been used to localize people.      
The ability to traverse walls and to bounce off human bodies in the WiFi range without carrying wireless transmitters is the major advantage for deploying a RF-based sensing system.      
Also, privacy can be preserved due to non-visual data.       
However, RF signals have a relatively low spatial resolution compared to visual camera images and the RF systems have
shown to generate coarse 3D pose estimation.     
Zhao et al. [234] proposed a RF-based deep learning method, named RFPose, to estimate 2D pose for multi-person scenarios. Later the extended version, named RF-Pose3D [235], can estimate 3D skeletons for multi-person.      
Based on these, Zhao et al. [236] presented a temporal adversarial training method with multi-headed attention module, named RF-Avatar, to recover a full 3D body mesh using the SMPL body model.       
Other sensors/sources: Besides using the aforementioned sensors, Isogawa et al. [237] estimated 3D human pose from the 3D spatio-temporal histogram of photons captured by a non-line-of-sight (NLOS) imaging system.     
Tome et al. [238] tackled the egocentric 3D pose estimation via a fish-eye camera.       
Saini et al. [239] estimated human motion using images captured by multiple Autonomous Micro Aerial Vehicles (MAVs).      
Clever et al. [240] focused on the HPE of the rest position in bed from pressure images which were collected by a pressure sensing mat.

ウェアラブルな慣性計測ユニット（IMU）は、物体のオクルージョンや衣服の障害物がない状態で動作を記録することで、特定の人体部位の向きや加速度を追跡することができます。     
しかし、IMUを使用する際には、時間経過とともにドリフト問題が発生する可能性があります。     
Marcardら[229]は，人体に取り付けられた6つのIMUから人体の姿勢を再構成するSparse Inertial Poser（SIP）を提案した．     
収集された情報は、正確な結果を得るために、コヒーレンス制約のあるSMPLボディモデルにフィッティングされた。    
Marcardら[230]は、さらに6-17個のIMUセンサと手持ちの移動カメラを関連付けて、in-the-wild 3D HPEを実現した。   
グラフベースの最適化手法が導入され、各2D人物検出を、長距離フレームからの3Dポーズ候補に割り当てることができました。     
Huang et al. [231] は，Sparse Inertial Poser (SIP) 法 [229] の限界に取り組んでいる．    
複数のポーズパラメータが同じIMUの姿勢を生成する可能性があり、また、IMUデータの収集には時間がかかります。
そこで，AMASSデータセット[176]のモーションキャプチャーシーケンスから方位と加速度を得るために，SMPLメッシュ上に仮想センサを配置して，大規模な合成データセットを作成しました．
また、IMUの姿勢と加速度を、過去と未来の情報を持つSMPLのパラメータにマッピングするために、双方向のRNNフレームワークが提案されました。      
Zhangら[232]は 姿勢正則化絵画構造モデルを導入して 姿勢正則化絵画構造モデルを導入し，IMUの姿勢に関連したマルチビュー・ヒートマップから3Dポーズを推定した．の姿勢を推定した。     
Huangら[233]は、IMUデータとマルチビュー画像を融合することで、2段階のアプローチであるDeepFuseを提案した。第1段階ではマルチビュー画像のみを処理してボリューム表現を予測し、第2段階ではIMUを用いてIMU-bone refinement layerにより3Dポーズを精密化する。     
無線周波数デバイス。無線周波数（RF）を利用したセンシング技術も、人の位置を特定するために利用されている。     
無線送信機を持ち歩かなくても、壁を通過したり、WiFi範囲内の人体に跳ね返ったりすることができるのは、RFベースのセンシングシステムを導入する上での大きなメリットです。     
また、非視覚的なデータのため、プライバシーを守ることができます。      
しかし、RF信号は、視覚的なカメラ画像に比べて空間分解能が低く、RFシステムは、粗い3Dポーズを生成することが示されています。
粗い3Dポーズ推定を行うことが示されている。    
Zhaoら[234]は、複数人のシナリオで2Dポーズを推定するために、RFPoseと名付けられたRFベースの深層学習法を提案した。その後、RF-Pose3D[235]と名付けられた拡張版は、多人数のための3Dスケルトンを推定することができる。     
これらに基づいて、Zhaoら[236]は、SMPLボディモデルを用いてフル3Dボディメッシュを復元するために、RF-Avatarと名付けられた多頭注意モジュールを用いた時間的敵対的学習法を発表した。      
その他のセンサー/ソース 前述のセンサ以外にも，Isogawaら[237]は，非線源（NLOS）イメージングシステムで撮影されたフォトンの3次元時空間ヒストグラムから人間の3次元ポーズを推定している．    
Tomeら[238]は、魚眼カメラを用いた自我中心の3Dポーズ推定に取り組んだ。      
Sainiら[239]は，複数の自律型小型航空機（MAV）で撮影した画像を用いて人間の動きを推定している．     
Cleverら[240]は、圧力感知マットによって収集された圧力画像からベッドでの休息位置のHPEに着目した。

#### 4.3 3D HPE Summary 3D HPEのまとめ
3D HPE has made significant advancements in recent years.      
Since a large number of 3D HPE methods apply the 2D to 3D lifting strategy, the performance of 3D HPE has been improved considerably due to the progress made in 2D HPE.     
Some 2D HPE methods such as OpenPose [17], CPN [90], AlphaPose [89], and HRNet [81] have been extensively used as 2D pose detector in 3D HPE methods.     
Besides the 3D pose, some methods also recover 3D human mesh from images or videos, e.g., [164] [175] [241] [242].     
However, despite the progress made so far, there are still several challenges.      
One challenge is the model generalization.      
High-quality 3D ground truth pose annotations depend on motion capture systems which cannot be easily deployed in random environment.      
Therefore, the existing datasets are mainly captured in constrained scenes.       
The state-of-the-art methods can achieve promising results on these datasets, but their performance degrades when applied to in-the-wild data.      
It is possible to leverage gaming engines to generate synthetic datasets with diverse poses and complex scenes, e.g., SURREAL dataset [174] and GTA-IM dataset [243].      
However, learning from synthetic data may not achieve the desired performance due to a gap between synthetic and real data distributions.       
Same as 2D HPE, robustness to occlusion and computation efficiency are two key challenges for 3D HPE as well.      
The performance of current 3D HPE methods drops considerably in crowded scenarios due to severe mutual occlusion and possibly low resolution content of each person.      
3D HPE is more computation demanding than 2D HPE.          
For example, 2D to 3D lifting approaches rely on 2D poses as intermediate representations for inferring 3D poses.       
Therefore, it is critical to develop computationally efficient 2D HPE pipelines while maintaining high accuracy for pose estimation. 

3D HPEは、近年大きく進歩している。     
多くの3D HPE手法は、2Dから3Dへのリフティング戦略を適用しているため、2D HPEの進歩により、3D HPEの性能が大幅に向上しています。    
OpenPose[17]、CPN[90]、AlphaPose[89]、HRNet[81]などのいくつかの2D HPE手法は、3D HPE手法の2Dポーズ検出器として広く使用されている。    
3D ポーズの他にも，画像やビデオから人間の 3D メッシュを復元する手法もある [164] [175] [241] [242]．    
しかし、これまでの進歩にもかかわらず、いくつかの課題が残っている。     
一つの課題は，モデルの一般化である．     
高品質の3Dグランドトゥルースポーズアノテーションは、ランダムな環境に容易に展開できないモーションキャプチャシステムに依存する。     
そのため、既存のデータセットは、主に制約のあるシーンで撮影されたものです。      
最先端の手法は、これらのデータセットでは有望な結果を得ることができますが、実世界のデータに適用すると、その性能は低下します。     
SURREALデータセット[174]やGTA-IMデータセット[243]などのように、ゲームエンジンを利用して、多様なポーズや複雑なシーンを持つ合成データセットを生成することが可能です。     
しかし、合成データからの学習では、合成データと実データの分布にギャップがあるため、期待した性能が得られない可能性がある。      
2D HPEと同様に、3D HPEにおいても、オクルージョンに対するロバスト性と計算効率が2つの重要な課題である。     
現在の3D HPE手法は、人混みの中では相互のオクルージョンがひどく、また各人のコンテンツが低解像度である可能性があるため、性能が大幅に低下します。     
3D HPEは、2D HPEよりも計算負荷が高い。         
例えば、2Dから3Dへのリフティング手法は、3Dポーズを推論するための中間表現として2Dポーズに依存している。      
そのため、高精度のポーズ推定を維持しつつ、計算効率の良い2D HPEパイプラインを開発することが重要です。

## 5 DATASETS AND EVALUATION METRICS データセットと評価指標
Datasets are very much needed in conducting HPE.      
They are also necessary to provide a fair comparison among different algorithms.       
Collecting a comprehensive and universal dataset poses challenges due to the complexity and variations of application scenes.       
A number of datasets have been collected to evaluate and compare results based on different metrics.     
In this section, we present the traditional datasets utilized in HPE, as well as more recent ones used for 2D and 3D deep learning-based HPE methods. 
In addition to these datasets with different features and task requirements, this section also covers several commonly used evaluation metrics for both 2D and 3D HPE. 
The results achieved by existing methods on the popular datasets are summarized as well.

データセットはHPEを行う上で非常に必要なものです。     
また、異なるアルゴリズム間で公正な比較を行うためにも必要です。      
しかし、アプリケーションシーンの複雑さと多様性のために、包括的で普遍的なデータセットを収集することは困難です。      
さまざまな評価基準に基づいて結果を評価・比較するために、多くのデータセットが収集されています。     
このセクションでは、HPEで利用される伝統的なデータセットと、2Dおよび3Dの深層学習ベースのHPE手法に利用されるより新しいデータセットを紹介します。    
特徴やタスク要件が異なるこれらのデータセットに加えて、このセクションでは、2Dおよび3D HPEの両方で一般的に使用されているいくつかの評価指標も取り上げています。   
また、一般的なデータセットで既存の手法が達成した結果についてもまとめています。

### 5.1 Datasets for 2D HPE 2D HPE用データセット
There are many 2D human pose datasets before deep learning found its way into human pose estimation.     
These datasets are of two types:      
(1) upper body pose datasets including Buffy Stickmen [244], ETHZ PASCAL Stickmen [245], We Are Family [246], Video Pose 2 [247].    
Activities [248]; and (2) full body pose datasets including PASCAL Person Layout [249], Sport [250] and UIUC people [251].        
However, only a few recent works use these 2D HPE datasets because they have many limitations such as lack of diverse object movements and small number of images.      
Since deep learning based approaches are fueled by large amounts of training data, only the large-scale 2D HPE datasets are reviewed in this section.      
They are summarized under two different categories (image-based and video-based) in Table 2.    

深層学習が人間の姿勢推定に使われるようになる前に、多くの2D人間の姿勢データセットがありました。    
これらのデータセットには2つのタイプがある。     
(1) Buffy Stickmen [244]、ETHZ PASCAL Stickmen [245]、We Are Family [246]、Video Pose 2 [247]、Activity [248]などの上半身のポーズデータセット。   
Activities [248]、（2）PASCAL Person Layout [249]、Sport [250]、UIUC people [251]などの全身のポーズのデータセット。       
しかし、これらの2D HPEデータセットは、多様な物体の動きがない、画像数が少ないなどの多くの制限があるため、最近ではわずかな作品しか使用されていない。     
深層学習に基づくアプローチは、大量の学習データによって促進されるため、このセクションでは、大規模な2D HPEデータセットのみをレビューする。     
これらのデータセットは、表2のように2つの異なるカテゴリ（画像ベースとビデオベース）にまとめられている。   

### 5.1.1 Image-based datasets 画像ベースのデータセット

#### Frames Labeled In Cinema (FLIC) Dataset
Frames Labeled In Cinema (FLIC) Dataset [252] is one of the early image-based 2D HPE datasets, which contains 5,003 images collected automatically from Hollywood movies.            
Around 4,000 images are used as the training set and the rest are used as the testing set.      
The FLIC dataset uses a body part detector named Poselets [260] to obtain about 20K person candidates from every tenth frame of 30 popular Hollywood movies.      
The subjects in these images have different kinds of poses.         
The full set of frames harvested from the movies is called the FLIC-full dataset.      
It is a superset of the original FLIC dataset and contains 20,928 occluded, non-frontal samples.       
A new FLIC-based dataset named FLIC-plus was introduced in [53] by removing all the images that contain the same scene with the test set in the FLIC dataset.     
Dataset Link: https://bensapp.github.io/flic-dataset.html

Frames Labeled In Cinema (FLIC) Dataset [252]は、初期の画像ベースの2D HPEデータセットの1つで、ハリウッド映画から自動的に収集された5,003枚の画像を含んでいます。           
約4,000枚の画像がトレーニングセットとして使用され、残りはテストセットとして使用されます。     
FLICデータセットでは，Poselets[260]と呼ばれる身体部位検出器を用いて，30本の人気ハリウッド映画の10フレームごとに，約2万人の人物候補を抽出しています．     
これらの画像に写っている被写体は，様々なポーズをとっています．        
映画から抽出されたフレームのフルセットは，FLIC-fullデータセットと呼ばれています．     
このデータセットは，オリジナルのFLICデータセットのスーパーセットであり，20,928個の非正面のオクルージョンサンプルを含んでいます．      
FLIC-plusと呼ばれる新しいFLICベースのデータセットは，FLICデータセットのテストセットと同じシーンを含むすべての画像を削除することによって，[53]で紹介されました．     
データセットのリンク： https://bensapp.github.io/flic-dataset.html

#### Leeds Sports Pose (LSP) Dataset  リーズ・スポーツ・ポーズ（LSP）データセット
Leeds Sports Pose (LSP) Dataset [16] has 2,000 annotated images from Flickr and 8 sports tags covering different sports including athletics, badminton, baseball, gymnastics, parkour, soccer, tennis, and volleyball.    
In the LSP dataset, every person’s full body is labeled with a total of 14 joints.      
In addition, the Leeds Sports Pose Extended dataset (LSPextended) [254] extends the LSP dataset and is only used for training.      
LSP-extended dataset has over 10,000 images from Flickr.      
In most recent research, LSP and LSP-extended datasets have been used for single-person HPE.       
Dataset Link: https://sam.johnson.io/research/lsp.html

Leeds Sports Pose (LSP) Dataset [16] は，Flickr からの 2,000 枚の注釈付き画像と，陸上競技，バトミントン，野球，体操，パルクール，サッカー，テニス，バレーボールなどのさまざまなスポーツをカバーする 8 つのスポーツタグを備えている．   
LSPデータセットでは、すべての人の全身に合計14の関節がラベル付けされている。     
さらに，Leeds Sports Pose Extended dataset (LSPextended) [254]は，LSPデータセットを拡張したもので，学習にのみ使用される．     
LSP-extendedデータセットは、Flickrから10,000枚以上の画像 の画像が含まれています。     
最近の研究では、LSPおよびLSP-extended データセットは，一人用のHPEに使用されている．      
データセットのリンク https://sam.johnson.io/research/lsp.html

#### Max Planck Institute for Informatics (MPII) Human Pose Dataset 
Max Planck Institute for Informatics (MPII) Human Pose Dataset [253] is a popular dataset for evaluation of articulated HPE.      
The dataset includes around 25,000 images containing over 40,000 individuals with 
annotated body joints.       
Based on [261], the images were systematically collected by a two-level hierarchical method to capture everyday human activities.       
The entire dataset covers 410 human activities and all the images are labeled.       
Each image was extracted from a YouTube video and provided with preceding and following un-annotated frames.       
Moreover, rich annotations including body part occlusions, 3D torso and head orientations are labeled by workers on Amazon Mechanical Turk.       
Images in MPII are suitable for 2D single-person or multi-person HPE. 
Dataset Link: http://human-pose.mpi-inf.mpg.de/#

Max Planck Institute for Informatics (MPII) Human Pose Dataset [253]は，関節型HPEの評価によく用いられるデータセットである．     
このデータセットには、40,000人以上の人物を含む約25,000枚の画像が含まれています。
このデータセットには，約25,000枚の画像が含まれている．      
261]に基づいて，画像は2レベルの階層的な方法で体系的に収集され，人間の日常的な活動を捉えている．      
データセット全体で410の人間の活動をカバーしており，すべての画像にはラベルが付けられている．      
各画像は，YouTubeの動画から抽出され，前後にアノテーションのないフレームが用意されている．      
さらに、Amazon Mechanical Turkの作業員によって、体の部位のオクルージョン、3Dの胴体や頭の向きなどの豊富なアノテーションが付けられている。      
MPIIの画像は、2Dの1人用または複数人用のHPEに適しています。
データセットリンク: http://human-pose.mpi-inf.mpg.de/#

#### Microsoft Common Objects in Context (COCO)
Microsoft Common Objects in Context (COCO) Dataset [108] is the most widely used large-scale dataset.      
It has more than 330,000 images and 200,000 labeled subjects with keypoints, and each individual person is labeled with 17 joints.      
The COCO dataset is not only proposed for pose estimation and analysis, but also used for object detection and image segmentation in natural environments, recognition in context, etc.      
There are two versions of the COCO datasets for HPE: COCO keypoints 2016 and COCO
keypoints 2017, which are hosted by COCO 2016 Keypoints Detection Challenge and COCO 2017 Keypoints Detection Challenge, respectively.     
The difference between COCO 2016 and COCO 2017 lies in the training, validation and test split.        
The COCO dataset has been extensively used in multiperson HPE works. 
In addition, Jin et al. [262] proposed COCO-WholeBody Dataset with whole-body annotations for HPE.       
Dataset Link: https://cocodataset.org/#home

Microsoft Common Objects in Context (COCO) Dataset [108]は，最も広く使われている大規模データセットである．     
このデータセットは，33万枚以上の画像と，キーポイントでラベル付けされた20万人の被写体を持ち，個々の人物には17個の関節がラベル付けされている．     
COCOデータセットは、ポーズ推定や解析のために提案されているだけでなく、自然環境下での物体検出や画像セグメンテーション、文脈の中での認識などにも利用されている。     
HPEのCOCOデータセットには、「COCO keypoints 2016」と「COCO
keypoints 2017の2種類があり、それぞれCOCO 2016 Keypoints Detection ChallengeとCOCO 2017 Keypoints Detection Challengeが主催しています。    
COCO 2016」と「COCO 2017」の違いは、トレーニング、検証、テストの分割にあります。       
COCOデータセットは、多人数のHPE作品で広く使用されています。
また、Jinら[262]は、HPEのために全身のアノテーションを施したCOCO-WholeBody Datasetを提案しました。      
データセットのリンク： https://cocodataset.org/#home

#### AI Challenger Human Keypoint Detection (AIC-HKD) Dataset
AI Challenger Human Keypoint Detection (AIC-HKD) Dataset [255] is currently the largest training dataset for 2D HPE.      
It has 300,000 annotated images for keypoint detection.      
There are 210,000 images for training, 30,000 images for validation and over 600,000 images for testing.     
The images were collected from internet search engines and mainly focused on daily activities of people. Dataset Link: https://challenger.ai/

AI Challenger Human Keypoint Detection (AIC-HKD) Dataset [255]は、現在、2D HPEの最大のトレーニングデータセットである。     
キーポイント検出のための30万枚のアノテーション画像があります。     
トレーニング用に210,000枚、検証用に30,000枚、テスト用に600,000枚以上の画像が用意されています。    
画像はインターネットの検索エンジンから収集したもので、主に人々の日常的な活動に焦点を当てています。データセットのリンク： https://challenger.ai/

#### CrowdPose Dataset
CrowdPose Dataset [256] is one of the latest dataset for 2D HPE in crowded and occlusive scenarios.     
This dataset contains 20,000 images selected from 30,000 images with Crowd Index (a measurement satisfies uniform distribution to judge the crowding level in images ).        
The training, validation and testing datasets have 10,000 images, 2,000 images and 8,000 images separately.      
Dataset Link: https://github.com/Jeff-sjtu/CrowdPose

CrowdPose データセット
CrowdPose Dataset [256]は、混雑した閉塞感のあるシナリオでの2D HPEのための最新データセットの一つです。    
このデータセットは，3万枚の画像から2万枚の画像を選び，Crowd Index（画像の混雑度を判断するための一様分布を満たす測定値）を用いています．       
また、トレーニングデータ、検証データ、テストデータには、それぞれ1万枚、2千枚、8千枚の画像が含まれています。     
データセットのリンク： https://github.com/Jeff-sjtu/CrowdPose

### 5.1.2 Video-based datasets ビデオベースのデータセット
#### Penn Action Dataset
Penn Action Dataset [257] consists of 2,326 video sequences with 15 different actions and human joint annotations.        
The videos contain frames with annotations from sports actions: baseball pitch, baseball swing, tennis forehand, tennis serve, bench press, bowling, clean and jerk, golf swing, jump rope, jumping jacks, pull up, push up, sit up, squat, and strum guitar.     
The annotations for images were labeled using Amazon Mechanical Turk.     
Dataset Link: http://dreamdragon.github.io/PennAction/

Penn Action Dataset [257] は，15 種類の動作と人間の関節のアノテーションを含む 2,326 のビデオシーケンスから構成される．       
動画には，野球の投球，野球のスイング，テニスのフォアハンド，テニスのサーブ，ベンチプレス，ボウリング，クリーン＆ジャーク，ゴルフのスイング，縄跳び，跳び箱，懸垂，腕立て伏せ，座位，スクワット，ギターのかき鳴らしなどのスポーツアクションのアノテーションが付いたフレームが含まれている．    
画像のアノテーションは，Amazon Mechanical Turkを使ってラベル付けしました．    
データセットのリンク： http://dreamdragon.github.io/PennAction/

#### Joint-annotated Human Motion Database
Joint-annotated Human Motion Database (J-HMDB) [258] is a fully annotated video dataset for action recognition, human detection and HPE.      
There are 21 action categories including brush hair, catch, clap, climb stairs, golf, jump, kick ball, pick, pour, pull-up, push, run, shoot ball, shoot bow, shoot gun, sit, stand, swing baseball, throw, walk, and wave.      
There are 928 video clips comprising 31,838 annotated frames.     
A 2D articulated human puppet model is applied to generate all the annotations based on Amazon Mechanical Turk.        
The 70 percent images in the J-HMDB dataset are used for training and the rest images are used for testing.       
Dataset Link: http://jhmdb.is.tue.mpg.de/

Joint-annotated Human Motion Database（J-HMDB） [258]は，行動認識，人間検出，HPE のための完全なアノテーション付きのビデオデータセットである．     
アクションには、髪をとかす、キャッチする、拍手する、階段を上る、ゴルフをする、ジャンプする、ボールを蹴る、拾う、注ぐ、引き上げる、押す、走る、ボールを撃つ、弓を撃つ、銃を撃つ、座る、立つ、野球のスイングをする、投げる、歩く、手を振るなど、21のカテゴリがあります。     
928個のビデオクリップ、31,838個のフレームにアノテーションが施されています。    
すべてのアノテーションの生成には，Amazon Mechanical Turkを利用した2次元関節人形モデルを適用した．       
J-HMDBデータセットの70%の画像を学習に、残りの画像をテストに使用しています。      
データセットのリンク： http://jhmdb.is.tue.mpg.de/

### PoseTrack Dataset
PoseTrack Dataset [259] is a large-scale dataset for multi- person pose estimation and articulated tracking in video analysis.       
Each person in a video has a unique track ID with annotations.      
PoseTrack contains 1,356 video sequences, around 46,000 annotated video frames and 276,000 body pose annotations for training, validation and testing.       
Dataset Link: https://posetrack.net/

PoseTrack Dataset [259]は，ビデオ解析における多人数のポーズ推定と関節追跡のための大規模なデータセットです．      
ビデオ内の各人物は，注釈付きのユニークなトラックIDを持っています．     
PoseTrackには、1,356のビデオシーケンス、約46,000のアノテーションされたビデオフレーム、276,000のボディポーズアノテーションがトレーニング、検証、テスト用に含まれています。      
データセットのリンク: https://posetrack.net/

### 5.2 Evaluation Metrics for 2D HPE 2D HPEの評価指標
It is difficult to precisely evaluate the performance of HPE because there are many features and requirements that need to be considered (e.g., upper/full human body, single/multiple pose estimation, the size of human body).       
As a result, many evaluation metrics have been used for 2D HPE.     
Here we summarize the commonly used ones.       

HPEの性能を正確に評価するのは難しい。なぜなら、考慮しなければならない特徴や要件が多いからである（例えば、上半身／全身、単一／複数のポーズ推定、人体の大きさ）。      
そのため、2D HPEには多くの評価指標が用いられている。    
ここでは、よく使われているものをまとめます。 

#### Percentage of Correct Parts(PCP) パーセンテージオブコレクトパーツ(PCP)
Percentage of Correct Parts (PCP) [263] is a measure commonly used in early works on 2D HPE, which evaluates stick predictions to report the localization accuracy for limbs.     
The localization of limbs is determined when the distance between the predicted joint and ground truth joint is less than a fraction of the limb length (between 0.1 to 0.5).      
In some works, the PCP measure is also referred to as PCP@0.5, where the threshold is 0.5. This measure is used on the LSP dataset for single-person HPE evaluation.      
However, PCP has not been widely implemented in latest works because it penalizes the limbs with short length which are hard to detect.      
The performance of a model is considered better when it has a higher PCP measure.     
In order to address the drawbacks of PCP, Percentage of Detected Joints (PDJ) is introduced, where a prediction joint is considered as detected if the distance between predicted joints and true joints is within a certain fraction of the torso diameter [36].     

Percentage of Correct Parts (PCP) [263] は、2D HPE の初期の作品でよく使われている指標で、スティック予測を評価して、手足のローカライズ精度を報告するものである。    
手足のローカライズは、予測されたジョイントとグランドトゥルースジョイントの間の距離が、手足の長さの何分の一か（0.1から0.5の間）よりも小さい場合に判断される。     
いくつかの作品では、PCPメジャーはPCP@0.5 とも呼ばれており、閾値は0.5であるとされている。この尺度は、一人用のHPE評価用のLSPデータセットで使用されています。     
しかし、PCPは検出が困難な長さの短い手足にペナルティを与えるため、最近の作品ではあまり実装されていません。     
また，PCPの値が大きいほど，モデルの性能が高いと考えられます．    
PCPの欠点を解決するために、Percentage of Detected Joints (PDJ)が導入され、予測関節と真の関節の間の距離が胴体の直径のある割合以内であれば、予測関節が検出されたとみなされる[36]。  

#### Percentage of Correct Keypoints (PCK)
Percentage of Correct Keypoints (PCK) [264] is also used to measure the accuracy of localization of different keypoints within a given threshold.      
The threshold is set to 50 percent of the head segment length of each test image and it is denoted as PCKh@0.5.      
PCK is referred to as PCK@0.2 when the distance between detected joints and true joints is less than 0.2 times the torso diameter.      
The higher the PCK value, the better model performance is regarded.      

また，Percentage of Correct Keypoints (PCK) [264]は，与えられた閾値内での異なるキーポイントのローカライズの精度を測定するために使用される．     
閾値は各テスト画像の頭部セグメント長の50％に設定され、PCKh@0.5 と表記されます。     
PCKは、検出された関節と真の関節の間の距離が、胴体の直径の0.2倍以下の場合に、PCK@0.2 と表記されます。     
PCKの値が大きいほど、モデルの性能が高いとみなされます。     

#### Average Precision (AP) and Average Recall (AR). 
AP measure is an index to measure the accuracy of keypoints detection according to precision (the ratio of true positive results to the total positive results) and recall (the ratio of true positive results to the total number of ground truth positives).       
AP computes the average precision value for recall over 0 to 1. AP has several similar variants.      
For example, Average Precision of Keypoints (APK) is introduced in [264].       
Mean Average Precision (mAP), which is the mean of average precision over all classes, is a widely used metric on the MPII and PoseTrack datasets. Average Recall (AR) is another metric used in the COCO keypoint evaluation [265].       
Object Keypoint Similarity (OKS) plays the similar role as the Intersection over Union (IoU) in object detection and is used for AP or AR.       
This measure is computed from the scale of the subject and the distance between predicted points and ground truth points.       
The COCO evaluation usually uses mAP across 10 OKS thresholds as the evaluation metric.

AP指標は，精度（正解数に対する真の正解数の割合）とリコール（グランドトゥルースの正解数に対する真の正解数の割合）に基づいて，キーポイント検出の精度を測定する指標です．      
APは，0から1にわたるrecallの平均精度値を計算する．APにはいくつかの類似した亜種がある．     
例えば，Average Precision of Keypoints (APK) は [264] で紹介されている．      
Mean Average Precision (mAP)は、全クラスの平均精度の平均であり、MPIIデータセットやPoseTrackデータセットで広く使用されている指標である。Average Recall (AR)は、COCOキーポイント評価で使用される別のメトリックである[265]。      
オブジェクトキーポイント類似度(OKS)は、オブジェクト検出におけるIoU(Intersection over Union)と同様の役割を果たし、APやARに使用される。      
この指標は、被写体のスケールと、予測点とグランドトゥルース点の距離から計算される。      
COCOの評価では、通常、10個のOKS閾値にわたるmAPを評価指標として使用しています。

### 5.3 Performance Comparison of 2D HPE Methods 2次元HPE法の性能比較
In Tables 3 ∼ 6, we have summarized the performance of different 2D HPE methods on the popular datasets together with the relevant and commonly used evaluation metrics.         
 For comparison on the LSP dataset, the PCP measure is employed to evaluate the performance of body detection-based and regression-based methods in Table 3.       
Table 4 shows the comparison results for different 2D HPE methods on the MPII dataset using PCKh@0.5 measure.      
It is worth noting that body detection methods generally have better performance than regression methods, thus gaining more popularity in recent 2D HPE research.      
In Table 5, the mAP comparison on the full testing set of the MPII dataset is reported.     
Table 6 presents the experimental results of different 2D HPE methods on the test-dev set of the COCO dataset, together with a summary of the experiment settings (extra data, backbones in models, input images size) and AP scores for each approach.     

表3〜表6では、一般的なデータセットにおけるさまざまな2D HPE手法の性能を、関連する一般的な評価指標とともにまとめています。        
 表3では，LSPデータセットでの比較のため，PCP指標を用いて，ボディ検出ベースの手法と回帰ベースの手法の性能を評価しています。      
表4は，MPIIデータセットにおける異なる2D HPE手法の比較結果を，PCKh@0.5 指標を用いて示したものである．     
ボディ検出法は一般的に回帰法よりも優れた性能を有しており、最近の2D HPE研究で人気を博していることは注目に値します。     
表5では、MPIIデータセットの全テストセットにおけるmAPの比較を報告しています。    
表6は、COCOデータセットのテスト-devセットにおけるさまざまな2D HPE手法の実験結果と、実験設定の概要（追加データ、モデル内のバックボーン、入力画像サイズ）、各手法のAPスコアを示しています。   

### 5.4 Datasets for 3D HPE 3D HPE 用データセット
In contrast to numerous 2D human pose datasets with highquality annotation, acquiring accurate 3D annotation for 3D HPE datasets is a challenging task that requires motion capture systems such as MoCap and wearable IMUs.      
Due to this requirement, many 3D pose datasets are created in constrained environments.      
Here, the widely used 3D pose datasets under different settings are summarized in Table 7.    

高品質のアノテーションが施された多数の2Dヒト・ポーズ・データセットとは対照的に、3D HPEデータセットのために正確な3Dアノテーションを取得することは、MoCapやウェアラブルIMUなどのモーションキャプチャシステムを必要とする困難な作業です。     
この要件のため、多くの3Dポーズデータセットは、制約のある環境で作成されています。     
ここでは、さまざまな環境下で広く使用されている3Dポーズデータセットを表7にまとめている。 

#### HumanEva Dataset HumanEvaデータセット
HumanEva Dataset [266] contains 7 calibrated video sequences (4 gray-scale and 3 color) with ground truth 3D annotation captured by a commercial MoCap system from ViconPeak.       
The database consists of 4 subjects performing 6 common actions (walking, jogging, gesturing, throwing and catching a ball, boxing, and combo) in a 3m × 2m area.       
Dataset Link: http://humaneva.is.tue.mpg.de/

HumanEva Dataset [266] には、ViconPeak社の商用MoCapシステムで撮影された、グラウンド・トゥルースの3Dアノテーション付きの7つのキャリブレーションされたビデオ・シーケンス（グレースケール4つ、カラー3つ）が含まれている。      
このデータベースは、4人の被験者が3m×2mのエリアで6つの一般的な動作（歩行、ジョギング、ジェスチャー、ボールを投げて捕まえる、ボクシング、コンボ）を行っています。      
データセットリンク： http://humaneva.is.tue.mpg.de/

#### Human3.6M ヒューマン3.6M
Human3.6M [267] is the most widely used indoor dataset for 3D HPE from monocular images and videos.        
There are 11 professional actors (6 males and 5 females) performing 17 activities (e.g., smoking, taking photo, talking on the phone) from 4 different views in an indoor laboratory environment.      
This dataset contains 3.6 million 3D human poses with 3D ground truth annotation captured by accurate marker-based MoCap system.      
There are 3 protocols with different training and testing data splits.       
Protocol #1 uses images of subjects S1, S5, S6, and S7 for training, and images of subjects S9 and S11 for testing.       
Protocol #2 uses the same training-testing split as Protocol #1, but the predictions are further post-processed by a rigid transformation before comparing to the ground-truth.          
Protocol #3 uses images of subjects S1, S5, S6, S7, and S9 for training, and images of subjects S11 for testing.     
Dataset Link: http://vision.imar.ro/human3.6m/

Human3.6M [267]は、単眼の画像やビデオからの3D HPEのための最も広く使用されている屋内データセットです。       
11人のプロの俳優（男性6人、女性5人）が、屋内の実験室環境で4つの異なる視点から17の活動（喫煙、写真撮影、電話での会話など）を行っています。     
このデータセットには、360万の3Dポーズと、正確なマーカーベースのMoCapシステムによって撮影された3Dグランドトゥルースアノテーションが含まれています。     
3つのプロトコルがあり、トレーニングデータとテストデータの分割が異なります。      
プロトコル#1では、被験者S1、S5、S6、S7の画像をトレーニングに、被験者S9とS11の画像をテストに使用しています。      
プロトコル#2は、プロトコル#1と同じトレーニングとテストの分割を使用していますが、予測値はグランドトゥルースと比較する前に、剛体変換によってさらに後処理されます。         
プロトコル#3では、トレーニングに被験者S1、S5、S6、S7、S9の画像を使用し、テストに被験者S11の画像を使用する。    
データセットリンク： http://vision.imar.ro/human3.6m/

#### MPI-INF-3DHP
MPI-INF-3DHP [269] is a dataset captured by a commercial marker-less MoCap system in a multi-camera studio.      
There are 8 actors (4 males and 4 females) performing 8 human activities including walking, sitting, complex exercise posed, and dynamic actions.      
More than 1.3 million frames from 14 cameras were recorded in a green screen studio
which allows automatic segmentation and augmentation.      
Dataset Link: http://gvv.mpi-inf.mpg.de/3dhp-dataset/ 

MPI-INF-3DHP [269]は，市販のマーカーレスMoCapシステムを用いて，マルチカメラスタジオで撮影されたデータセットである．     
8人のアクター（男性4人、女性4人）が、歩く、座る、複雑な運動のポーズ、ダイナミックなアクションなど、8つのヒューマンアクティビティを行っている。     
14台のカメラから130万以上のフレームをグリーンスクリーンスタジオで記録しました。
これにより、自動的なセグメンテーションとオーグメンテーションが可能になりました。     
データセットのリンク: http://gvv.mpi-inf.mpg.de/3dhp-dataset/ 

#### TotalCapture Dataset
TotalCapture Dataset [270] contains fully synchronised videos with IMU and Vicon labeling for over 1.9 million frames.     
There are 13 sensors placed on key body parts such as head, upper and lower back, upper and lower limbs, and feet.      
The data was collected indoors with 8 calibrated full HD video cameras at 60 Hz measuring roughly 4 × 6 m.     
There are 5 actors (4 males and 1 female) performing actions, repeated 3 times, including walking, running, and freestyle.      
Dataset Link: https://cvssp.org/data/totalcapture/

TotalCapture Dataset [270]は、190万フレーム以上のIMUおよびViconラベリングされた完全に同期したビデオを含んでいます。    
頭部、背中の上部と下部、手足の上部と下部、足など、体の主要部分に13個のセンサーが配置されています。     
データは屋内で、キャリブレーションされた8台のフルHDビデオカメラを用いて、60Hzで約4×6mの大きさで収集されました。    
5人のアクター（男性4人、女性1人）が、歩く、走る、フリースタイルなどの動作を3回繰り返しています。     
データセットのリンク： https://cvssp.org/data/totalcapture/

#### CMU Panoptic Dataset
CMU Panoptic Dataset [268] contains 65 sequences (5.5 hours) with 1.5 million of 3D skeletons of multiple people scenes.      
This dataset was captured by a marker-less motion capture system with 480 VGA camera views, more than 30 HD views, 10 RGB-D sensors, and a calibrated hardwarebased synchronization system.       
The test set contains 9,600 frames from HD cameras for 4 activities (Ultimatum, Mafia, Haggling, and Pizza).      
Dataset Link: domedb.perception.cs.cmu.edu/

CMU Panoptic Dataset [268]には，65個のシーケンス（5.5時間）と150万個の複数の人物シーンの3Dスケルトンが含まれています．     
このデータセットは，480個のVGAカメラビュー，30個以上のHDビュー，10個のRGB-Dセンサ，およびキャリブレーションされたハードウェアベースの同期システムを備えたマーカーレスモーションキャプチャシステムによってキャプチャされています．      
テストセットには、4つのアクティビティ（Ultimatum, Mafia, Haggling, Pizza）のHDカメラからの9,600フレームが含まれています。     
データセットリンク: domedb.perception.cs.cmu.edu/

#### 3DPW Dataset
3DPW Dataset [230] was collected by hand-held cameras with IMUs in natural scenes capturing daily activities (e.g., shopping in the city, going up-stairs, doing sports, drinking coffee, and taking the bus).      
There are 60 video sequences (more than 51,000 frames) in this dataset and the corresponding 3D poses were computed by wearable IMUs.     
The test set contains 9,600 frames from HD cameras for 4 activities (Ultimatum, Mafia, Haggling, and Pizza).      
Dataset Link: https://virtualhumans.mpi-inf.mpg.de/3DPW/

3DPWデータセット[230]は，IMUを搭載したハンドヘルドカメラを用いて，日常的な行動（街での買い物，階段を上る，スポーツをする，コーヒーを飲む，バスに乗るなど）を撮影した自然なシーンを収集したものである．     
このデータセットには60のビデオシーケンス（51,000フレーム以上）が含まれており，対応する3DポーズはウェアラブルIMUによって計算されています．    
テストセットには、4つのアクティビティ（Ultimatum, Mafia, Haggling, Pizza）に対するHDカメラからの9,600フレームが含まれています。     
データセットのリンク： https://virtualhumans.mpi-inf.mpg.de/3DPW/

#### MuCo-3DHP
MuCo-3DHP Dataset [197] is a multi-person 3D training set composed by the MPI-INF-3DHP single-person dataset with ground truth 3D pose from multi-view marker-less motion capture system.      
Background augmentation and shading-aware foreground augmentation of person appearance were applied to enable data diversity.   
Dataset Link: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/

MuCo-3DHP Dataset [197]は，MPI-INF-3DHPの1人用データセットに，多視点マーカーレスモーションキャプチャシステムから得られたグランドトゥルースの3Dポーズを加えた，複数人用の3Dトレーニングセットです．     
データの多様性を実現するために，人物の外見の背景拡張とシェーディングを考慮した前景拡張が適用されています．  
データセットのリンク: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/

#### MuPoTS-3D Dataset
MuPoTS-3D Dataset [197] is a multi-person 3D test set and its ground-truth 3D poses were captured by a multi-view marker-less MoCap system containing 20 real-world scenes (5 indoor and 15 outdoor).     
There are challenging samples with occlusions, drastic illumination changes, and lens flares in some of the outdoor footage.      
More than 8,000 frames were collected in the 20 sequences by 8 subjects. 
Dataset Link: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/

MuPoTS-3D Dataset [197]は、多人数用の3Dテストセットで、20の実世界シーン（屋内5、屋外15）を含むマルチビューマーカーレスMoCapシステムで撮影された3Dポーズのグランドトゥルースです。    
屋外の映像の一部には、オクルージョン、急激な照明変化、レンズフレアなどの困難なサンプルがあります。     
8人の被験者により、20シーンで8,000以上のフレームが収集されました。
データセットリンク: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/

#### AMASS Dataset
AMASS Dataset [176] was created by unifying 15 different optical marker-based MoCap datasets and using the SMPL model to represent human motion sequences.     
This large dataset contains more than 40 hours of motion data in 8,593 sequences of 9 million frames sampled at 60 Hz. More than 11,000 motions were recorded over 300 subjects.       
Dataset Link: https://amass.is.tue.mpg.de/

AMASSデータセット[176]は、15の異なる光学マーカーベースのMoCapデータセットを統合し、SMPLモデルを用いて人間のモーションシーケンスを表現することで作成されました。    
この大規模なデータセットには、60Hzでサンプリングされた900万フレームの8,593シーケンスの40時間以上のモーションデータが含まれています。300人の被験者に対して、11,000以上のモーションが記録されています。      
データセットのリンク： https://amass.is.tue.mpg.de/

NBA2K
NBA2K Dataset [271] was extracted from the NBA2K19 video games by intercepting calls between the game engine and the graphic card using RenderDoc.     
The synthetic dataset contains 27,144 basketball poses spanning 27 subjects.      
The 3D poses of 35 keypoints and the corresponding RGB images are provided in this dataset with high quality.     
Dataset Link:https://github.com/luyangzhu/NBA2K-dataset

NBA2K Dataset [271]は，RenderDocを用いてゲームエンジンとグラフィックカード間のコールを傍受することで，NBA2K19ビデオゲームから抽出されたものである．    
この合成データセットには，27人の被験者のバスケットボールのポーズが27,144個含まれています．     
このデータセットでは，35個のキーポイントの3Dポーズと，それに対応するRGB画像が高品質で提供されています．    
データセットのリンク:https://github.com/luyangzhu/NBA2K-dataset

#### GTA-IM 
GTA-IM Dataset [243] is a GTA Indoor Motion dataset collected from Grand Theft Auto (GTA) video game by the GTA game engine.    
There are one million RGB-D frames of 1920 × 1080 resolution with ground-truth 3D human pose of 98 joints, covering various actions including sitting, walking, climbing, and opening the door.       
Each scene contains several settings such as living rooms, bedrooms and kitchens that emphasize human-scene interactions.      
Dataset Link: https://people.eecs.berkeley.edu/∼zhecao/hmp/

GTA-IMデータセット[243]は，ビデオゲーム「Grand Theft Auto（GTA）」からGTAゲームエンジンによって収集されたGTAインドアモーションデータセットである．   
このデータセットは，1920 × 1080 の解像度の 100 万個の RGB-D フレームと，98 個の関節からなるグランドトゥルースな 3D 人体姿勢で構成されており，座る，歩く，登る，ドアを開けるなどの様々な動作をカバーしています．      
各シーンには、リビングルーム、ベッドルーム、キッチンなどの設定があり、人間とシーンのインタラクションが強調されています。     
データセットリンク： https://people.eecs.berkeley.edu/∼zhecao/hmp/

#### Occlusion-Person Dataset オクルージョン人物データセット
Occlusion-Person Dataset [213] is a multi-view synthetic dataset with occlusion labels for the joints in images. UnrealCV [272] was used to render multi-view images and depth maps from 3D models.      
A total of 8 cameras were used every 45 degrees on a circle of two meters radius.     
This dataset contains 73K frames with 20.3% of the joints occluded.       
The ground truth 3D annotation and occlusion labels are also provided.       
Dataset Link: https://github.com/zhezh/occlusion_person

Occlusion-Person Dataset [213] は，画像中の関節に対するオクルージョン・ラベルを含むマルチビュー合成データセットです．マルチビュー画像と3Dモデルのデプスマップのレンダリングには、UnrealCV [272]が使用されました。     
半径2メートルの円の上に、45度ごとに合計8台のカメラを使用しました。    
このデータセットには73Kフレームが含まれており，関節の20.3%がオクルージョンしている．      
また、グランドトゥルースの3Dアノテーションとオクルージョンラベルも提供されています。      
データセットのリンク: https://github.com/zhezh/occlusion_person

### 5.5 Evaluation Metrics for 3D HPE 3D HPEの評価指標
#### MPJPE (Mean Per Joint Position Error) 
MPJPE (Mean Per Joint Position Error) is the most widely used evaluation metric to assess the performance of 3D HPE.     
MPJPE is computed by using the Euclidean distance between the estimated 3D joints and the ground truth positions as follows:      
where N is the number of joints, Ji and J∗i are the ground truth position and the estimated position of the ith joint, respectively.     
PMPJPE, also called Reconstruction Error, is the MPJPE after rigid alignment by a post-processing between the estimated pose and the ground truth pose.      
NMPJPE is defined as the MPJPE after normalizing the predicted positions in scale to the reference [205].      

MPJPE（Mean Per Joint Position Error）は，3D HPEの性能を評価するために最も広く用いられている評価指標です．    
MPJPEは，推定された3次元関節とグランドトゥルースの位置との間のユークリッド距離を用いて，以下のように計算される．     
ここで，Nは関節の数です．JiおよびJ∗iは，それぞれi番目の関節のGround Truth位置と推定位置である．    
PMPJPEはReconstruction Errorとも呼ばれ，推定ポーズとグランドトゥルースポーズの間に後処理を施してリジッドアライメントを行った後のMPJPEです．     
NMPJPEは、予測された位置を基準にスケールで正規化した後のMPJPEと定義される[205]。   

#### MPVE (Mean Per Vertex Error)
MPVE (Mean Per Vertex Error) [169] measures the Euclidean distances between the ground truth vertices and the predicted vertices as follows:
where N is the number of vertices, V is the ground truth vertices, and V∗ is the estimated vertices.

MPVE（Mean Per Vertex Error）[169]は，以下のように，グランドトゥルースの頂点と予測された頂点の間のユークリッド距離を測定する．
ここで，N は頂点の数，V はグランドトゥルースの頂点，V∗ は推定された頂点である．

#### 3DPCK
3DPCK is a 3D extended version of the Percentage of Correct Keypoints (PCK) metric used in 2D HPE evaluation.     
An estimated joint is considered as correct if the distance between the estimation and the ground-truth is within a certain threshold.       
Generally the threshold is set to 150mm.   

3DPCKは，2D HPE評価で用いられているPCK（Percentage of Correct Keypoints）メトリクスを3Dに拡張したものである。    
推定されたジョイントは、推定値とグランドトゥルースの間の距離がある閾値以内であれば、正しいとみなされます。      
一般的に、この閾値は150mmに設定されています。

### Summary.  まとめ 
As pointed out by Ji et al. [273], low MPJPE does not always indicate an accurate pose estimation as it depends on the predicted scale of human shape and skeleton.       
Although 3DPCK is more robust to incorrect joints, it cannot evaluate the precision of correct joints. Also, existing metrics are designed to evaluate the precision of an estimated pose in a single frame.       
However, the temporal consistency and smoothness of reconstructed human pose cannot be examined over continuous frames by existing evaluation metrics.     
Designing frame-level evaluation metrics that can evaluate 3D HPE performance with temporal consistency and smoothness remains an open problem.   

Jiら[273]が指摘しているように、MPJPEの低さは、人間の形状や骨格の予測スケールに依存するため、必ずしも正確なポーズ推定を示しているとは限らない。      
3DPCKは正しくない関節に対してより頑健であるが、正しい関節の精度を評価することはできない。また、既存の評価指標は、1つのフレームで推定されたポーズの精度を評価するように設計されている。      
しかし、既存の評価指標では、再構成された人間のポーズの時間的な一貫性や滑らかさを連続したフレームで調べることができません。    
時間的な整合性や滑らかさを考慮して、3D HPE の性能を評価できるフレームレベルの評価指標を設計することは、今後の課題です。  

### 5.6 Performance Comparison of 3D HPE Methods 3D HPE手法の性能比較
Tables 8 ∼ 11 provide performance comparison of different 3D HPE methods on the widely used datasets corresponding to the single-view sing le-person, single-view multi-person, and multi-view scenarios.     
In Table 8, most 3D single-view single-person HPE models successfully estimate 3D human pose on the Human3.6M dataset with remarkable precision.     
Although the Human3.6M dataset has a large size of training and testing data, it only contains 11 actors (6 male and 5 female) performing 17 activities such as eating, discussion, smoking, and taking photo.       
When estimating 3D pose on the in-the-wild data with more complex scenarios, the performance of these methods are degraded.      
It is also observed that model-based methods perform on par with model-free methods.     
Utilizing temporal information when video data is available can improve the performance.     
3D single-view multi-person HPE is a harder task than 3D single-view single-person HPE due to more severe occlusion.      
As shown in Table 9 and Table 10, good progress has been made in single-view multi-person HPE methods in recent years.      
By comparing the results from Table 8 and Table 11, it is evident that the performance (e.g., MPJPE under Protocol 1) of multi-view 3D HPE methods has improved compared to single-view 3D HPE methods using the same dataset and evaluation metric.      

表8〜表11は、広く使われているデータセットを対象に、3D HPE手法の性能比較を示したもので、単視点1人、単視点複数人、複数視点のシナリオに対応している。    
表8では、ほとんどの3D単視点1人用HPEモデルが、Human3.6Mデータセットにおいて、人間の3Dポーズを非常に高い精度で推定することに成功しています。    
Human3.6Mデータセットは、トレーニングデータとテストデータのサイズが大きいにもかかわらず、11人のアクター（男性6人、女性5人）が、食事、議論、喫煙、写真撮影など17のアクティビティを行っているだけです。      
より複雑なシナリオを持つ実世界のデータに対して3Dポーズを推定する場合、これらの手法の性能は低下してしまう。     
また，モデルベースの手法は，モデルフリーの手法と同等の性能を示すことが確認された．    
ビデオデータが利用可能な場合は、時間情報を利用することで、性能を向上させることができます。    
3D単視点多人数HPEは、3D単視点単人数HPEよりもオクルージョンが激しく、難しいタスクです。     
表9および表10に示すように，近年，単視点多人数HPE手法は大きく進歩している。     
表8と表11の結果を比較すると、同じデータセットと評価指標を用いた場合、多視点3D HPE手法の性能（例えば、プロトコル1のMPJPE）が単視点3D HPE手法に比べて向上していることがわかります。    

#### TABLE 8: 
Comparison of different 3D single-view single-person HPE approaches on the Human3.6M dataset.      
The best two scores are marked in red and blue, respectively.     
Here in model-free approaches, “Direct” indicates the method directly estimating 3D pose without 2D pose representation.     
“Lifting” indicates the method lifting the 2D pose representation to the 3D space (i.e., 3D pose). 
“Temporal” means the method using temporal information.

Human3.6Mデータセットにおける，異なる3Dシングルビュー1人用HPE手法の比較．     
ベスト2のスコアはそれぞれ赤と青で示されている。    
ここで、モデルフリーアプローチにおいて、"Direct "は、2Dポーズ表現を用いずに直接3Dポーズを推定する手法を示す。    
"Lifting "は、2Dのポーズ表現を3D空間（すなわち3Dポーズ）に持ち上げる手法を示す。
"Temporal "は、時間情報を利用する手法を示す。

#### TABLE 9: 
Comparison of different 3D single-view multiperson HPE approaches on the MuPoTS-3D dataset.     
The best two scores are marked in red and blue, respectively.     

MuPoTS-3Dデータセットにおける3Dシングルビュー・マルチパーソンHPEアプローチの比較。    
ベスト2のスコアはそれぞれ赤と青で示されている。  

#### TABLE 10: 
Comparison of different 3D single-view multiperson HPE approaches on the CMU Panoptic dataset.      
Ultimatum, Mafia, Haggling, and pizza denote four activities.       
The best two scores are marked in red and blue, respectively.

CMU Panopticデータセットにおける3Dシングルビュー・マルチパーソンHPEアプローチの比較。     
Ultimatum, Mafia, Haggling, pizzaは4つのアクティビティを表す。      
ベスト2のスコアはそれぞれ赤と青で示されている。   

#### TABLE 11: 
Comparison of different 3D multi-view HPE approaches on the Human3.6M dataset.     
The best two scores are marked in red and blue, respectively.   

Human3.6Mデータセットにおける異なる3DマルチビューHPEアプローチの比較。    
ベスト2のスコアはそれぞれ赤と青で示されている。  

### 5.7 Conference Workshops and Challenges for HPE HPEに関する会議のワークショップとチャレンジ
Due to the increasing interest in HPE, workshops and challenges on HPE are held in conjunction with computer vision conference venues like CVPR, ICCV and ECCV.     
These workshops aim at gathering researchers and practitioners work on HPE to discuss the current state-of-the-art as well as future directions to concentrate on.     
In Table 12, we summarize the relevant 2D and 3D HPE workshops and challenges in this research field from 2017 to 2020.

HPEへの関心の高まりを受けて、CVPR、ICCV、ECCVなどのコンピュータビジョン会議の会場では、HPEに関するワークショップやチャレンジが開催されています。    
これらのワークショップは、HPEに取り組む研究者や実務者を集めて、現在の最新技術や今後注力すべき方向性について議論することを目的としている。    
表12では、2017年から2020年にかけて、この研究分野に関連する2Dおよび3DのHPEワークショップと課題をまとめている。

ICCV 2017 PoseTrack Challenge: Human Pose Estimation and Tracking in the Wild https://posetrack.net/workshops/iccv2017/    
ICCV 2017 PoseTrack Challenge：野生での人間のポーズ推定と追跡 https://posetrack.net/workshops/iccv2017/

ECCV 2018  PoseTrack Challenge: Articulated People Tracking in the Wild  https://posetrack.net/workshops/eccv2018/

CVPR 2018  3D humans 2018: 1st International workshop on Human pose, motion, activities and shape  https://project.inria.fr/humans2018/#

CVPR 2019  3D humans 2019: 2nd International workshop on Human pose, motion, activities and shape  https://sites.google.com/view/humans3d/

CVPR 2019  Workshop On Augmented Human: Human-centric Understanding  https://vuhcs.github.io/vuhcs- 2019/index.html

CVPR 2020  Towards Human-Centric Image/Video Synthesis  https://vuhcs.github.io/

ECCV 2020  3D poses in the wild challenge  https://virtualhumans.mpi-inf.mpg.de/3DPW Challenge/

ACM Multimedia 2020  Large-scale Human-centric Video Analysis in Complex Events http://humaninevents.org/


## 6 APPLICATIONS 応用例
In this section, we review related works of exploring HPE for a few popular applications (Fig. 7).

このセクションでは、いくつかの一般的なアプリケーションのためにHPEを探索した関連作品をレビューします（図7）。

### Action recognition, prediction, detection, and tracking:      行動認識、予測、検出、追跡    
Pose information has been utilized as cues for various applications such as action recognition, prediction, detection, and tracking.       
Angelini et al. [274] proposed a real-time action recognition method using a pose-based algorithm.       
Yan et al. [275] leveraged the dynamic skeleton modality of pose for action recognition.          
Markovitz et al. [276] studied human pose graphs for anomaly detection of human actions in videos.      
Cao et al. [243] used the predicted 3D pose for long-term human motion prediction.       
Sun et al. [277] proposed a viewinvariant probabilistic pose embedding for video alignment.     
Pose-based video surveillance enjoys the advantage of preserving privacy by monitoring through pose and human mesh representation instead of human sensitive identities.
Das et al. [278] embedded video with pose to identify activities of daily living for monitoring human behavior.

姿勢情報は、行動認識、予測、検出、追跡などの様々なアプリケーションの手掛かりとして利用されている。      
Angeliniら[274]は，姿勢ベースのアルゴリズムを用いたリアルタイム行動認識手法を提案している．      
Yanら[275]は、行動認識のためにポーズの動的スケルトンモダリティを活用した。         
Markovitz ら[276]は、動画内の人間の行動を異常に検出するための人間のポーズ・グラフを研究している。     
Cao ら [243] は、予測された 3D ポーズを長期的な人間の動きの予測に利用した。      
Sunら[277]は、ビデオの位置合わせのために、視野不変の確率的ポーズ埋め込みを提案した。    
ポーズベースのビデオ監視は、人間の敏感なアイデンティティではなく、ポーズと人間のメッシュ表現を通して監視することで、プライバシーを保護できるという利点があります。
Dasら[278]は、人間の行動を監視するために、日常生活の活動を識別するために、ビデオにポーズを埋め込んだ。

### Action correction and online coaching: アクション補正とオンラインコーチング。
Some activities such as dancing, sporting, and professional training require precise human body control to strictly react as the standard pose.      
Normally personal trainers are responsible for the pose correction and action guidance in a face-to-face manner.       
With the help of 3D HPE and action detection, AI personal trainers can make coaching more convenient by simply setting up cameras without personal trainer presenting.       
Wang et al. [279] designed an AI coaching system with a pose estimation module for personalized athletic training assistance.      

ダンスやスポーツ、プロのトレーニングなどのアクティビティでは、基準となるポーズに厳密に反応するために、人間の正確なボディコントロールが求められます。     
通常、パーソナルトレーナーは、対面でポーズ補正や動作指導を行います。      
3D HPEと行動検出を利用すれば、パーソナルトレーナーが提示しなくても、カメラを設置するだけで、AIパーソナルトレーナーがコーチングをより便利に行うことができます。      
Wangら[279]は、パーソナライズされたアスレチックトレーニング支援のために、ポーズ推定モジュールを備えたAIコーチングシステムを設計しました。  

### Clothes parsing: 洋服のパーシング 
The e-commerce trends have brought about a noticeable impact on various aspects including clothes purchases. 
Clothing product in pictures can no longer satisfy customers’ demands, and customers hope to see the reliable appearance as they wear their selected clothes.
Clothes parsing and pose transfer [280] make it possible by inferring the 3D appearance of a person wearing a specific clothes. 
HPE can provide plausible human body regions for cloth parsing. 
Moreover, the recommendation system can be upgraded by evaluating appropriateness based on the inferred reliable 3D appearance of customers with selected items. 
Patel et al. [281] achieved clothing prediction from 3D pose, shape and garment style.

電子商取引の動向は、衣服の購入を含む様々な面で顕著な影響をもたらしています。
写真に写っている服は，もはや顧客の要求を満足させるものではなく，顧客は選んだ服を着たときの信頼できる姿を見たいと思っている．
それを可能にするのが Clothes parsing and pose transfer [280] であり，特定の服を着ている人の3次元的な外観を推論することができる．
HPEは、服の解析のために、もっともらしい人体領域を提供することができる。
さらに、選択したアイテムを身につけた顧客の推定された信頼できる3D外観に基づいて適切性を評価することで、推薦システムをアップグレードすることができます。
Patelら[281]は、3Dのポーズ、形状、衣服のスタイルから衣服の予測を実現した。

### Animation, movie, and gaming: アニメーション、映画、ゲーム。
Motion capture is the key component to present characters with complex movements and realistic physical interactions in industries of animation, movie, and gaming.     
The motion capture devices are usually expensive and complicated to set up.     
HPE can provide realistic pose information while alleviating the demand for professional high-cost equipment [282] [283].

モーションキャプチャーは、アニメーション、映画、ゲームなどの業界で、複雑な動きやリアルな身体的インタラクションを持つキャラクターを表現するための重要な要素です。    
モーションキャプチャー機器は通常、高価でセットアップも複雑です。    
HPEは、プロフェッショナルな高額機器の需要を軽減しながら、リアルなポーズ情報を提供することができる [282] [283]。 

### AR and VR: 
Augmented Reality (AR) technology aims to enhance the interactive experience of digital objects into the real-world environment.      
The objective of Virtual Reality (VR) technology is to provide an immersive experience for the users.      
AR and VR devices use human pose information as input to achieve their goals of different applications.      
A cartoon character can be generated in real-world scenes to replace the real person. 
Weng et al. [284] created 3D character animation from single photo with the help of 3D pose estimation and human mesh recovery.      
Zhang et al. [285] presented a pose-based system that converts broadcast tennis match videos into interactive and controllable video sprites.     
The players in the video sprites preserve the techniques and styles as real professional players.

AR（Augmented Reality：拡張現実）技術は、デジタルオブジェクトを現実世界の環境に取り込み、インタラクティブな体験を強化することを目的としています。     
バーチャルリアリティ（VR）技術は、ユーザーに没入感を与えることを目的としています。     
ARおよびVRデバイスは、さまざまなアプリケーションの目的を達成するために、人間のポーズ情報を入力として使用します。     
実世界のシーンでは、実在の人物の代わりに漫画のキャラクターを生成することができる。
Wengら[284]は，3Dポーズ推定と人間のメッシュ復元を利用して，1枚の写真から3Dキャラクター・アニメーションを作成した．     
Zhang ら[285]は，放送されたテニスの試合映像を，インタラクテ ィブで制御可能なビデオ・スプライトに変換するポーズベースのシステ ムを発表した．    
ビデオ・スプライトの中の選手は，実際のプロ選手と同じようなテクニックやスタイルを維持している．

### Healthcare: ヘルスケア 
HPE provides quantitative human motion information that physicians can diagnose some complex diseases, create rehabilitation training, and operate physical therapy.      
Lu et al. [286] designed a pose-based estimation system for assessing Parkinson’s disease motor severity.      
Gu et al. [287] developed a pose-based physical therapy system that patients can be evaluated and advised at home.      
Furthermore, such a system can be established to detect abnormal action and to predict the following actions ahead of time.      
Alerts are sent immediately if the system determines that danger may occur.      
Chen et al. [288] used the HPE algorithms for fall detection monitoring in order to provide immediate assistant.      
Also, HPE methods can provide reliable posture labels of patients in hospital environments to augment research on neural correlates to natural behaviors [289].    
HPE は定量的な人間の動作情報を提供するため、医師は複雑な疾患の診断、リハビリテーショントレーニングの作成、理学療法の実施が可能になります。     
Lu ら [286]は、パーキンソン病の運動強度を評価するための、姿勢に基づいた推定システムを設計しました。     
Gu ら [287]は、患者が自宅で評価やアドバイスを受けられるように、姿勢ベースの理学療法システムを開発しました。     
さらに、このようなシステムは、異常な行動を検出し、次の行動を事前に予測するために構築することができます。     
危険が発生する可能性があるとシステムが判断した場合には、直ちに警告が送られる。     
Chenら[288]は、転倒検知モニタリングにHPEアルゴリズムを使用して、即時のアシスタントを提供している。     
また、HPE手法は、病院環境にいる患者の信頼できる姿勢ラベルを提供し、自然な行動の神経相関に関する研究を強化することができる[289]。

## 7 CONCLUSION AND FUTURE DIRECTIONS 結論と将来の方向性
In this survey, we have presented a systematic overview of recent deep learning-based 2D and 3D HPE methods.       
A comprehensive taxonomy and performance comparison of these methods have been covered.       
Despite great success, there are still many challenges as discussed in Sections 3.3 and 4.3. Here, we further point out a few promising future directions to promote advances in HPE research.

この調査では、最近の深層学習ベースの2Dおよび3D HPE手法の体系的な概要を紹介しました。      
これらの手法の包括的な分類法と性能比較を取り上げた。      
大きな成功にもかかわらず、セクション3.3と4.3で議論したように、まだ多くの課題があります。ここではさらに、HPE研究の進歩を促進するために、いくつかの有望な将来の方向性を指摘する。

### Domain adaptation for HPE.      HPEのドメイン適応。     
For some applications such as estimating human pose from infant images [290] or artwork collections [291], there are not enough training data with ground truth annotations.       
Moreover, data for these applications exhibit different distributions from that of the standard pose datasets.     
HPE methods trained on existing standard datasets may not generalize well across different domains.      
The recent trend to alleviate the domain gap is utilizing GAN-based learning approaches.       
Nonetheless, how to effectively transfer the human pose knowledge to bridge domain gaps remains unaddressed.    

乳児画像からの人間の姿勢の推定[290]やアートワークコレクション[291]などの一部のアプリケーションでは、グランドトゥルースアノテーションを含む十分なトレーニングデータがない。      
さらに、これらのアプリケーションのデータは、標準的なポーズデータセットのデータとは異なる分布を示している。    
既存の標準的なデータセットで学習したHPE手法は、異なるドメイン間でうまく一般化できない可能性がある。     
ドメインギャップを軽減するための最近の傾向は、GANベースの学習アプローチを利用することです。      
しかし、ドメインギャップを埋めるために、人間のポーズの知識をどのように効果的に伝達するかは、まだ解決されていない。 

### 3D model shape and pose
Human body models such as SMPL, SMPLify, SMPLX, GHUM & GHUML, and Adam are used to model human mesh representation.     
However, these models have a huge number of parameters.      
How to reduce the number of parameters while preserving the reconstructed mesh quality is an intriguing problem.      
Also, different people have various deformations of body shape.      
A more effective human body model may utilize other information such as BMI [180] and silhouette [292] for better generalization.     

人体のメッシュ表現をモデル化するために、SMPL、SMPLify、SMPLX、GHUM & GHUML、Adamなどの人体モデルが使われています。    
しかし、これらのモデルは、膨大な数のパラメータを持っています。     
再構成されたメッシュの品質を保ちつつ、いかにしてパラメータの数を減らすかは、興味のある問題です。     
また、人によって体の形は様々に変形します。     
より効果的な人体モデルは，BMI[180]やシルエット[292]などの他の情報を利用して，より良い一般化を図ることができるかもしれない．

### Most existing methods ignore human interaction with 3D scenes.  既存の手法の多くは、3Dシーンにおける人間のインタラクションを無視しています。
There are strong human-scene relationship 20 constraints that can be explored such as a human subject cannot be simultaneously present in the locations of other objects in the scene.      
The physical constraints with semantic cues can provide reliable and realistic 3D HPE.

例えば、人間はシーン内の他のオブジェクトの位置に同時に存在することはできないなど、人間とシーンの強い関係20の制約を探ることができます。     
物理的な制約と意味的な手がかりを組み合わせることで、信頼性の高い、リアルな3D HPEを実現できます。

### 3D HPE is employed in visual tracking and analysis.     3D HPEは、視覚的な追跡や分析に採用されています。
Existing 3D human pose and shape reconstruction from videos are not smooth and continuous.     
One reason is that the evaluation metrics such as MPJPE cannot evaluate the smoothness and the degree of realisticness.      
Appropriate frame-level evaluation metrics focusing on temporal consistency and motion smoothness should be developed.  
    
既存の動画からの3次元人物姿勢・形状復元は、滑らかで連続的ではありません。    
その理由の一つは、MPJPE などの評価指標では、滑らかさやリアルさの度合いを評価できないためです。     
時間的な整合性や動きの滑らかさに着目した適切なフレームレベルの評価指標を開発する必要がある。    

### Existing well-trained networks pay less attention to resolution mismatch.  既存のよく訓練されたネットワークは、解像度の不一致にはあまり注意を払いません。
The training data of HPE networks are usually high resolution images or videos, which may lead to inaccurate estimation when predicting human pose from low resolution input.      
The contrastive learning scheme [293] (e.g., the original image and its low resolution version as a positive pair) might be helpful for building resolutionaware HPE networks.   

HPEネットワークの学習データは、通常、高解像度の画像や動画であり、低解像度の入力から人間の姿勢を予測する場合には、不正確な推定につながる可能性がある。     
解像度を考慮したHPEネットワークを構築するためには、コントラスト学習スキーム[293]（例えば、元の画像とその低解像度バージョンを正のペアとする）が役立つかもしれない。    

### Deep neural networks in vision tasks are vulnerable to adversarial attacks.   ビジョンタスクにおけるディープニューラルネットワークは、敵対的な攻撃に対して脆弱です   
The imperceptible noise can significantly affect the performance of HPE.     
There are few works [294] [295] that consider adversarial attack for HPE.     
The study of defense against adversarial attacks can improve the robustness of HPE networks and facilitate real-world pose-based applications.

感知できないノイズは、HPEの性能に大きな影響を与える。    
HPEに対する敵対的攻撃を考慮した研究はほとんどない [294] [295]。    
敵対的攻撃に対する防御の研究は、HPE ネットワークのロバスト性を向上させ、実世界でのポーズベースのアプリケーションを促進することができる。

### Human body parts may have different movement patterns and shapes due to the heterogeneity of the human body.    人間の体のパーツは、その不均質性ゆえに、異なる動きのパターンや形状を持つことがあります。     
A single shared network architecture may not be optimal for estimating all body parts with various degrees of freedom.     
Neural Architecture Search (NAS) [296] can search the optimal architecture for estimating each body part [297], [298].     
Also, NAS can be used for discovering efficient HPE network architectures to reduce the computational cost [299].     
It is also worth exploring multi-objective NAS in HPE when multiple objectives (e.g, latency, accuracy and energy consumption) have to be met.

さまざまな自由度を持つすべての体の部位を推定するためには、単一の共有ネットワーク・アーキテクチャでは最適ではない場合があります。    
Neural Architecture Search (NAS) [296]は，各身体部位を推定するための最適なアーキテクチャを探索することができる [297]，[298]．    
また、NASは、計算コストを削減するために、効率的なHPEネットワーク・アーキテクチャを発見するために使用することができる[299]。    
複数の目的（レイテンシー、精度、エネルギー消費など）を満たさなければならない場合、HPEにおける多目的NASを調査することも価値がある。
