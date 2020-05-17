# GUI-RPD-Discounting-Observation Errorsの説明
繰り返し囚人のジレンマゲームの戦略が，割引因子や観測エラーによってどのような影響を受けるのか，このツールを使うことによって，理解を深めることができる．

- <img src="https://latex.codecogs.com/gif.latex?{\bf%20p}=(p_1,p_2,p_3,p_4),p_0"> は自分（縦軸）の戦略
- <img src="https://latex.codecogs.com/gif.latex?\epsilon,\xi">はそれぞれ一方のみがエラーする確率，両方エラーする確率
- <img src="https://latex.codecogs.com/gif.latex?w">は割引因子　（w=1のときは割引なし）
- 相手（横軸）の戦略 <img src="https://latex.codecogs.com/gif.latex?{\bf%20q}">はランダム

自分の戦略を1つに決め，ランダムに決めた相手の1,000+2戦略に対して，それぞれゲームを行った時の利得関係を表す．
相手の+2戦略分は，ALLC戦略（青点）とALLD（赤点）である．
## 操作方法
### スライダー
- 計算方法の切り替え
  - (0)Detは本研究で見つけた行列式で計算する方法，(1)はHilbe et al.,2015,GEBで示された逆行列の形で計算する方法である
- 割引因子<img src="https://latex.codecogs.com/gif.latex?w">，エラー率<img src="https://latex.codecogs.com/gif.latex?\epsilon,\xi">は，スライダーで変更できる
- 戦略<img src="https://latex.codecogs.com/gif.latex?{\bf%20p}=(p_1,p_2,p_3,p_4),p_0=1">の値もスライダーから変更できる．1,000分の1単位で値を変えることができる．
- 横のzdボタンを押すと，ZD戦略となる値に自動調整してくれる（ZD戦略となる値がある場合のみ）
### ボタン
- Other Opponent
  - 相手の戦略<img src="https://latex.codecogs.com/gif.latex?{\bf%20q}">の値を別のランダムな値に変更する．
- TRPS 5310
  - <img src="https://latex.codecogs.com/gif.latex?(T,R,P,S)=(1.5,1,0,-0.5)">から<img src="https://latex.codecogs.com/gif.latex?(T,R,P,S)=(5,3,1,0)">に変更する．
- Save Fig
  - 任意の場所に今表示している図をPNG形式で保存する．
- Switch Normal/AP
  - 通常モードと偏微分係数モードを切り替える．
- Quit
  - ツールを終了する．

### 偏微分係数モード
自分 (相手) の利得<img src="https://latex.codecogs.com/gif.latex?s_X"> <img src="https://latex.codecogs.com/gif.latex?(s_Y)">について，相手の戦略<img src="https://latex.codecogs.com/gif.latex?{\bf%20q}=(q_1,q_2,q_3,q_4),q_0">による偏微分係数の値: <img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_X/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i"> (<img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_Y/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i">) を計算し，各点について色でその大小を表示する．値の計算はChen & Zinger, 2014, JTBの式(13)の右辺を参考にした．

偏微分係数の値の意味は次のように考えられる．
* 値が正: 協力率を上げると利得が高くなる
* 値が負: 協力率を上げると利得が低くなる

#### 専用リストボックス
* <img src="https://latex.codecogs.com/gif.latex?q_i,(i=0,1,2,3,4)">のボタン
  - それぞれについての偏微分の値を切り替える．Blue: 正，Red: 負（Whiteはゼロ付近）．
* suboptimal
  - <img src="https://latex.codecogs.com/gif.latex?i=0,...,4">について<img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_X/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i"> (<img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_Y/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i">) の二乗和を取り，その値を表示する．ゼロだと赤になり，それ以外では白点で表示される（この値がゼロ=そのときの<img src="https://latex.codecogs.com/gif.latex?{\bf%20q}">は停留点となり，戦略変更の動機付けを持たない）
  
#### スライダー
カラーマップの最大・最小値の絶対値を決める（小さくするほど0に近い偏微分の値が色に現れやすくなる）


### ショートカットキー
* p:
  - 通常モードと偏微分係数モードを切り替える（Switch Nomal/APボタンと同じ機能）．
* Shift+p:
  - 偏微分係数を計算する際，プレイヤーXの利得についての偏微分（<img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_X/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i">) かプレイヤーYの利得の偏微分 (<img src="https://latex.codecogs.com/gif.latex?\partial"><img src="https://latex.codecogs.com/gif.latex?s_Y/\partial"><img src="https://latex.codecogs.com/gif.latex?q_i">) かを切り替える．
* Shift+c:
  - 相手の戦略<img src="https://latex.codecogs.com/gif.latex?{\bf%20q}">の値をコーナーケースに設定する．

## UIの例
### 条件
- <img src="https://latex.codecogs.com/gif.latex?(T,R,P,S)=(1.5,1,0,-0.5)">
- <img src="https://latex.codecogs.com/gif.latex?w=1">　（割引なし）
- <img src="https://latex.codecogs.com/gif.latex?%28%5Cepsilon%2C%5Cxi%29%3D%280%2C0%29">　（エラーなし）
- <img src="https://latex.codecogs.com/gif.latex?{\bf%20p}=(1,0,0,1),%20p_0=1"> （WSLS戦略）

※相手がALLCのときは，定常分布がなく計算できないので，青点が表示されない．

![wsls strategy](https://github.com/azm17/RPD/blob/master/wsls.PNG "wsls")
