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

## INTRODUCTION
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

## Previous surveys and our contributions
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
