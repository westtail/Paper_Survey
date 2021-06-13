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
