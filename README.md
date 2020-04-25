医師・医学博士。人工知能 x バイオインフォマティクス x 合成生物学によるイノベーションとその医学応用を目指す研究者です (これまでの研究については<a href='https://researchmap.jp/h_shimizu'>こちら</a>)。
このレポジトリーでは有用なリソースや学術論文を忘備録的にまとめています。
<h1>人工知能</h1>
<h2>GitHubレポジトリー</h2>
<a href='https://github.com/danielecook/Awesome-Bioinformatics'>Awesome Bioinformatics</a>: バイオインフォに関するキュレーションされたソフト・ライブラリー・リソースのリスト<br>
<a href='https://github.com/hsiaoyi0504/awesome-cheminformatics'>Awesome Cheminformatics</a>: ケモインフォに関するキュレーションされたソフト・ライブラリー・リソースのリスト<br>
<a href='https://github.com/sdmg15/Best-websites-a-programmer-should-visit'>Best-websites-a-programmer-should-visit</a>: プログラミングに関する有用なウェブサイトへのリンク集<br>
<a href='https://github.com/microsoft/forecasting'>Time Series Forecasting Best Practices & Examples</a>: Microsoftがまとめた時系列モデル構築のベストプラクティス集。Python/R双方のJupyter Notebookつき。

<h2>読みかけ論文</h2>
<a href='https://www.nature.com/articles/s41586-020-2188-x'>A reference map of the human binary protein interactome</a>
その<a href='https://github.com/CCSB-DFCI/HuRI_paper'>コード</a>


<h2>機械学習・深層学習のフレームワーク</h2>
Scikit-Learn<br>
<a href='https://pycaret.org/'>PyCaret</a>: データ前処理や可視化、機械学習のモデル開発を数行のコードで出来てしまうPythonのライブラリ
TensorFlow<br>
Keras<br>
PyTorch<br>
<a href='https://github.com/pytorch/serve'>TorchServe</a>: PyTorchモデルをREST API化するツール。Amazon & Facebook共同開発<br>
<a href='https://github.com/deepchem/deepchem'>DeepChem</a>: 化学の深層学習フレームワーク<br>

<h2>自然言語処理</h2>
<a href='https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp'>DistilBERT-base-jp</a>:日本語Wikipediaで学習したDistilBERT。MITライセンス。<br>
<a href='https://towardsdatascience.com/natural-language-processing-the-age-of-transformers-a36c0265937d'>Natural Language Processing: the Age of Transformers</a>自然言語処理の短い歴史 (RNN, Attention,Transformer)

<h2>強化学習</h2>
<a href='https://github.com/Baekalfen/PyBoy'>PyBoy</a>:Python製のゲームボーイエミュレーター

<h2>機械学習・深層学習のツール</h2>
<a href='https://openai.com/blog/microscope/'>OpenAI Microscope</a>: OpenAIが開発した、代表的なDNNモデル内のノード関係を視覚的にみられるサイト。フィルターやノードが何に反応するか(活性されるか)が可視化されている。

<h2>バイオインフォマティクス情報源</h2>
<a href='http://togotv.dbcls.jp/'>TogoTV</a><br>
<a href='https://www.biostars.org/'>Biostars</a><br>


<h2>機械学習の学術論文・資料</h2>
<h3>ニューラルレンダリング</h3>
<a href='https://arxiv.org/abs/2004.03805?utm_campaign=piqcy&utm_medium=email&utm_source=Revue%20newsletter'>State of the Art on Neural Rendering</a> (2020) DNNを利用した画像描画に関する研究。CGで使われる物理法則、画像生成処理をそれぞれ紹介した後に、実際のレンダリング手法を様々な観点から評価を行った。Table1は要チェック。
<h3>アルゴリズム自動生成</h3>
<a href='https://arxiv.org/abs/2004.03805'>AutoML-Zero: Evolving Machine Learning Algorithms From Scratch</a> (2020) 機械学習アルゴリズムを自動生成するAutoML Zeroの研究。内部のロジックは高校レベルの演算の組み合わせ、探索は進化戦略。誤差逆伝搬で学習する2層のネットワークを発見できた<a href='https://github.com/google-research/google-research/tree/master/automl_zero'>コード</a> (C++) あり<br>
<a href='https://petar-v.com/talks/Algo-WWW.pdf'>Graph Representation Learning for Algorithmic Reasoning</a> DeepMind社の資料で、Graph Neural Networkを使用してアルゴリズムを学習/発見する研究の紹介。演算の実行、ステップレベルの演算学習、End2Endでのアルゴリズム学習という進化を提示している。

<h3>物体検出</h3>
<a href='https://arxiv.org/abs/2004.10934'>YOLOv4: Optimal Speed and Accuracy of Object Detection</a> (2020) YOLOv4の報告。

<h3>時系列データ処理</h3>
<a href='https://speakerdeck.com/motokimura/shi-jian-karumanhuiruta'>実践カルマンフィルタ</a> 時系列で変化する状態を推定するカルマンフィルタの解説スライド。カルマンフィルタはロボットなどの自己位置推定に使用されており、KITTI(自動運転車)のデータセットでその有効性を確認している(実験コードあり)。

<h3>強化学習</h3>
<a href='https://speakerdeck.com/motokimura/shi-jian-karumanhuiruta'>An Optimistic Perspective on Offline Reinforcement Learning</a> 学習済みエージェントの行動履歴から学習するOffline強化学習についてのGoogleの研究。新しいデータが取れない状態で汎化させるため、複数エージェントの価値予測をランダムにアンサンブルして予測を行う(Random Ensemble Mixture)。強化学習版の蒸留だ。
<h1>バイオインフォマティクス</h1>
<h2>バイオインフォマティクスデータベース</h2>

<h3>タンパク質</h3>
<a href='https://www.ebi.ac.uk/intact/'>IntAct</a>: EBIが運営するタンパク-タンパク結合実験データベース<br>
<a href='https://www.ebi.ac.uk/complexportal/home'>Complex Portal</a>: EBIが運営するマニュアルキュレーションしたタンパク複合体のデータベース
<h3>創薬</h3>
<a href='https://www.drugbank.ca/'>DrugBank</a>: 治療薬とその標的遺伝子に関するデータベース<br>
<a href='http://db.idrblab.net/ttd/'>TTD (Therapeutic Targets Database)</a>: 治療薬とその標的遺伝子に関するデータベース<br>
<a href='http://insilico.charite.de/supertarget/'>SuperTarget</a>: 治療薬とその標的遺伝子に関するデータベース、特に標的遺伝子のデータが充実<br>
<a href='https://www.genome.jp/kegg-bin/get_htext?br08303'>ATC (Anatomical Therapeutic Chemical) Classification</a>: 解剖治療化学分類法という、WHO分類に基づく疾患と治療薬の網羅的対応表<br>
<a href='http://stitch.embl.de/'>STITCH</a>: 化合部とタンパク質が構成するネットワーク情報のウェブサイト<br>
<a href='https://string-db.org/'>STRING</a>: タンパク質間相互作用ネットワーク情報のウェブサイト<br>

<h1>合成生物学</h1>
