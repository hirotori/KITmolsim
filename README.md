# KITMolSim

[En](#English) / [Ja](#Japanese)
<a id="English"></a>

KITmolsim is a python package of utility modules for molecular simulations.
- various particles (cube, crystal sphere, icosphere, etc.)
- modules for analyzing (center-of-mass of cluster, gyration tensor, etc.)
- initializing (randomize particle position, etc.)
- others (mpcd property calculator, etc.)
  
## Prerequisities
The list below is the python packages with which we assured this package correctly works:
- numpy >= 1.24.4, less than 2.~
- scipy >= 1.11.1
- scikit-learn >= 1.3.0
- numba >= 0.57.1

The `numba` package is not strictly necessary. If not found, this package uses the Fortran library (found on`kitmolsim/analyze/flib`) instead.

## Install

move this package to the directory where PYTHONPATH exists.

Install required libraries if they have not yet been installed.

```bash
pip3 install numpy scipy scikit-learn numba
```

If needed, build fortran modules in `flib` by using `make`.

```bash
make
```

<a id="Japanese"></a>
(Ja)

KITmolsimは分子シミュレーションに役立つモジュールをまとめたPythonパッケージです. 
- 様々な粒子形状 (立方体, fccなどの結晶球, ico球など)
- 後処理に使う解析用の関数
- 初期構造作成に便利な関数
- その他

## 必要なパッケージ
以下のリストは正常に動作が確認できているパッケージです:
- numpy >= 1.24.4 (ただしver.2.~未満．ver.2の互換性は未検証.)
- scipy >= 1.11.1
- scikit-learn >= 1.3.0
- numba >= 0.57.1

これらのうち`numba`は無くても良いです. 見つからない場合, 自動的に内部のFortranライブラリを用います. 

## インストール
このパッケージをPythonのパスが通っているディレクトリへ移動してください. 

必要なパッケージがインストールされていない場合は適宜インストールしてください. 

もし必要なら, `flib`内のFortranモジュールをビルドしてください．`make`コマンドが必要です．

```bash
make
```
