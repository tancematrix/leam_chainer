# 20180521
## prezto
 “prezto”で検索すると本家のGithubが出るのでそれに従って.zshrcに追記すれば良い。カスタマイズ->　https://dev.classmethod.jp/tool/zsh-prezto/
 システム環境設定からキー入力のリピート間隔を短縮するとdelとかカーソルの移動とかが速くなって快適
## GPU環境構築
 研究室wikiからやれば良い。virtualenvは、開発中の環境によってライブラリのバージョンを分けたりするのに大事。
 `Python3/`に`chainer`, `gensim`, `cupy=cuda80`を入れた
## sshfsエラー
 `ps`コマンドで走っているプロセスを確認できる。今回はリモートレポジトリとの接続プロセスが切れていなかった。
 `$ ps | grep PROCESS_NAME` で欲しいプロセスを抜き出せる
 Killコマンドでプロセスを殺す`$ kill -QUIT PROCESS_ID`
## クリアコマンド
　Ctr-Lでコマンドラインが一番上に上がる（`clear`と叩いても同じ）
## less
 `less`コマンドで、ファイルを一画面分だけ上から読み込んで表示できる。一気に表示すると長すぎるファイルなどに。
## IPython
 ソースコード内に
```
 from IPython import embed; embed()
```
 と書き込むことで、処理をそこで止め、そこまでに現れた変数の確認などをpythonのインタラクティブモードみたいな感じで行うことができる。
## 例外処理
```
try:
    hogehoge
exept:
    # 例外処理
```
で例外処理が書ける。
## dect.get()
`dict.get('key', x)` と書くと、`key`が辞書内にあるときはそのindexを、ないときは`x`を返す。
## arrayのブロードキャスト
 ベクトルをブロードキャストする
　`vec.shape`が`(SIZE,)`のとき、`vec[:,NONE]`とするとshapeが`(SIZE,1)`になってブロードキャストが効く
## padding
