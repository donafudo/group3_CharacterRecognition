# ひらがなの手書き文字認識

## 概要
あるひらがな1文字が書かれた画像に対して文字認識を行い、その画像に書かれている文字の判別を行う

## 環境


## 開発者
+ 下地 剛史     <e185428@ie.u-ryukyu.ac.jp>
+ 多和田 真都 <e185761@ie.u-ryukyu.ac.jp>
+ 藤渕 はな     <e185767@ie.u-ryukyu.ac.jp>


## 実行方法
このプロジェクトのトップディレクトリで`python3 ./src/model_test [画像ファイルのパス]`を実行すると選択した画像ファイルのひらがな分類を行う。


## ディレクトリ構成

- 機械学習の実装コード
`CharacterRecognition/src`

- テストデータ
`CharacterRecognition/testImgs`

- ドキュメント
`CharacterRecognition/docs/_build/index.html`

## HowTo
新たにモデルのパラメータを変更した状態で学習を行いたい場合の手順
コマンド、pythonファイルは全てプロジェクトのルートディレクトリから実行
1. [ETL文字データベース](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database)から圧縮ファイルを入手
1. 入手したファイルを解凍してディレクトリごと`CharacterRecognition/datasets/`に移動させる
1. load_etl.pyのread_etl()を実行。`CharacterRecognition/extract`以下に読み込まれた画像データが保存される。
1. cnn.pyに記述されているモデルのパラメータを直接書き換える
1. cnnのインスタンスを作成training()を実行
