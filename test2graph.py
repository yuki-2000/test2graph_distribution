import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression



#実験条件

#試験片幅[mm]
width = [14.89,14.90,14.89,14.90,14.91,14.90,14.92,14.92,14.93] 
print("平均幅",np.mean(width)) 
#試験片厚さ[mm]
thickness = [1.29,1.29,1.29,1.26,1.27,1.26,1.29,1.29,1.29]
print("平均厚さ",np.mean(thickness)) 
#断面積[m^2]
area = np.mean(width) * np.mean(thickness) * 10**-6
print("断面積[m^2]", area)



#ひずみゲージのゲージ率
gauge_rate = 2.1

#[N/V]荷重の出力電圧が何Nに対応するか
Force_scale = 50000 / 5 #50000Nが5Vに対応

#[mm/V]クロスヘッド変位の出力電圧が何mmに対応するか
cross_head_scale = 50 / 5 #50mmが5Vに対応






#データの読み込み
#header=56はcsvの最初のほうの行を無視している　
#skipfooterはCSVの最後のスキップする行数（適切に指定しないと数字が文字列になってしまう。　また、これを使うためにengine='python'としている）
df = pd.read_csv('test_result1.csv',encoding="shift jis",header=56, skipfooter=3, engine='python')
print(df.dtypes)



#なぜか数字が文字列になったとき用にfloatに変換
#for j in range(2,6):
    #df.iloc[:,j] = df.iloc[:,j].astype(float)








#データの変換

#応力[MPa]
df["stress"] = df["(1)HA-V01"] * Force_scale / (area * 10**6)
#クロスヘッド変位[mm]
df["cross_head_displacement"] = df["(1)HA-V02"] * cross_head_scale

#ひずみ[-]マイクロでもなく%でもなく無次元
df["strainL1"] = df["(2)ST-CH01"]*2*10**-6/gauge_rate
df["strainT1"] = df["(2)ST-CH02"]*2*10**-6/gauge_rate
df["strainL2"] = df["(2)ST-CH03"]*2*10**-6/gauge_rate
df["strainT2"] = df["(2)ST-CH04"]*2*10**-6/gauge_rate

#ひずみの裏表平均
df["strainLave"] = (df["strainL1"] + df["strainL2"])/2
df["strainTave"] = (df["strainT1"] + df["strainT2"])/2

#ポワソン比のひずみ変化
df["poissonLT"] = -1 *df["strainTave"] / df["strainLave"]








#最大応力表示
#https://www.self-study-blog.com/dokugaku/python-pandas-dataframe-max-min-describe/
#https://www.yutaka-note.com/entry/pandas_maxmin
#最大。最小・平均などをまとめて表示
df_des = df.describe()
print(df_des)


stress_max_index = df["stress"].idxmax()
print("最大応力[MPa]", stress_max_index, "番目　値", df["stress"][stress_max_index])



#弾性率計算
#ひずみ0.0005と0.0025における引張り応力を使用して弾性率を最小二乗法で回帰する

#i=1から初めて0.0005を超えたら破断後などのデータが混ざらない
for i in range(len(df)):
    if df["strainLave"][i] > 0.0005:
        print("i=",i,"ひずみ[-]",df["strainLave"][i])
        start_reg_index = i
        break
#i=1から初めて0.0025を超えたら破断後などのデータが混ざらない
for i in range(len(df)):
    if df["strainLave"][i] > 0.0025:
        print("i=",i, "ひずみ[-]",df["strainLave"][i])
        end_reg_index = i
        break


# 最小二乗法モデルで予測式を求める
#https://qiita.com/niwasawa/items/400afeb5239e197bb53f
#https://laid-back-scientist.com/least-squares
#xに関しては、[[10.0], [8.0], [13.0]]というようにしないといけないのでreshape
model = LinearRegression()
model.fit(np.array(df["strainLave"][start_reg_index : end_reg_index]).reshape(-1, 1) , df["stress"][start_reg_index : end_reg_index])
#print("弾性率について、傾きが弾性率[MPa]")
#print('切片:', model.intercept_)
#print('傾き:', model.coef_[0])
print("弾性率[MPa]",model.coef_[0])




#ひずみ0.0005と0.0025における縦ひずみ横ひずみを使用してポワソン比を最小二乗法で回帰する
poisson_model = LinearRegression()
poisson_model.fit(np.array(df["strainLave"][start_reg_index : end_reg_index]).reshape(-1, 1) , df["strainTave"][start_reg_index : end_reg_index])
#print("ポワソン比について、マイナス傾きがポワソン比[-]")
#print('切片:', poisson_model.intercept_)
#print('傾き:', poisson_model.coef_[0])
print("ポワソン比[-]",-poisson_model.coef_[0])








#グラフ描画
#https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
#https://qiita.com/Nick_utuiuc/items/9bf839f5612c54606348
#https://phst.hateblo.jp/entry/2020/02/28/000000
#https://qiita.com/M_Kumagai/items/b11de7c9d06b3c43431d


#グラフ設定

#フォント設定
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
#plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
#plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
#plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます



#軸設定
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
#plt.rcParams['axes.grid'] = True # make grid
#plt.rcParams['grid.linestyle']='--' #グリッドの線種
plt.rcParams["xtick.minor.visible"] = True  #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True  #y軸補助目盛りの追加
plt.rcParams['xtick.top'] = True                   #x軸の上部目盛り
plt.rcParams['ytick.right'] = True                 #y軸の右部目盛り



#軸大きさ
#plt.rcParams["xtick.major.width"] = 1.0             #x軸主目盛り線の線幅
#plt.rcParams["ytick.major.width"] = 1.0             #y軸主目盛り線の線幅
#plt.rcParams["xtick.minor.width"] = 1.0             #x軸補助目盛り線の線幅
#plt.rcParams["ytick.minor.width"] = 1.0             #y軸補助目盛り線の線幅
#plt.rcParams["xtick.major.size"] = 10               #x軸主目盛り線の長さ
#plt.rcParams["ytick.major.size"] = 10               #y軸主目盛り線の長さ
#plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
#plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
#plt.rcParams["axes.linewidth"] = 1.0                #囲みの太さ




#凡例設定
plt.rcParams["legend.fancybox"] = False  # 丸角OFF
plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
plt.rcParams["legend.markerscale"] = 5 #markerサイズの倍率














#グラフ描画テンプレ

strain_names = ["strainL1","strainL2","strainLave","strainT1","strainT2","strainTave"]
for col_name in strain_names:
    plt.figure(figsize=(7,5),dpi=300)
    plt.plot(df[col_name],df["stress"], label=col_name,c="k",lw=1)
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [MPa]")
    plt.legend()
    #plt.title("Template")
    plt.show()



#応力ひずみ線図 L方向まとめ
plt.figure(figsize=(7,5),dpi=300)
plt.plot(df["strainL1"],df["stress"], label="strainL1",lw=1)
plt.plot(df["strainL2"],df["stress"], label="strainL2",lw=1)
plt.plot(df["strainLave"],df["stress"], label="strainLave",lw=1)
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
plt.legend()
#plt.title("Template")
plt.show()





#応力ひずみ線図 T方向まとめ
plt.figure(figsize=(7,5),dpi=300)
plt.plot(df["strainT1"],df["stress"], label="strainT1",lw=1)
plt.plot(df["strainT2"],df["stress"], label="strainT2",lw=1)
plt.plot(df["strainTave"],df["stress"], label="strainTave",lw=1)
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
plt.legend()
#plt.title("Template")
plt.show()



#ポワソン比
plt.figure(figsize=(7,5),dpi=300)
plt.plot(df["strainLave"],df["strainTave"], label="poissonLT",c="k",lw=1)
plt.xlabel("StrainL [-]")
plt.ylabel("StrainT [-]")
plt.legend()
#plt.title("Template")
plt.show()

#ポワソン比の縦ひずみ変化
plt.figure(figsize=(7,5),dpi=300)
plt.plot(df["strainLave"],df["poissonLT"], label="PoissonLT",c="k",lw=1)
plt.xlabel("StrainL [-]")
plt.ylabel("PoissonLT")
plt.legend()
#plt.title("Template")
plt.show()














#弾性率のグラフ
x = np.arange(0.0005, 0.0025, 0.00001)
y = model.predict(x.reshape(-1, 1))

# 可視化 ひずみ0.0005と0.0025における
plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainLave"][start_reg_index : end_reg_index] , df["stress"][start_reg_index : end_reg_index], s=1, c='k')
plt.plot(x, y, c='r')
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
#plt.title("Template")
plt.show()

# 可視化　全体
plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainLave"] , df["stress"], s=1, c='k')
plt.plot(x, y, c='r')
plt.xlabel("Strain [-]")
plt.ylabel("Stress [MPa]")
#plt.title("Template")
plt.show()








#ポアソン比のグラフ
x = np.arange(0.0005, 0.0025, 0.00001)
y = poisson_model.predict(x.reshape(-1, 1))

# 可視化 ひずみ0.0005と0.0025における
plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainLave"][start_reg_index : end_reg_index] , df["strainTave"][start_reg_index : end_reg_index], s=1, c='k')
plt.plot(x, y, c='r')
plt.xlabel("StrainL [-]")
plt.ylabel("StrainT [-]")
#plt.title("Template")
plt.show()

# 可視化　全体
plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainLave"] , df["strainTave"], s=1, c='k')
plt.plot(x, y, c='r')
plt.xlabel("StrainL [-]")
plt.ylabel("StrainT [-]")
#plt.title("Template")
plt.show()


























#---------------------------------------------------------
#論文用
#範囲、軸指定、色指定など










#応力ひずみ線図 L方向

plt.figure(figsize=(7,5),dpi=300)
plt.plot(df["strainLave"],df["stress"], label="strainLave",lw=1,c="k")
plt.xlabel("StrainL [-]")
plt.ylabel("Stress [MPa]")

#軸の範囲
plt.xlim(0,0.020)
plt.ylim(0, 60)

#軸の指定
plt.xticks(np.arange(0, 0.021, 0.005))
plt.yticks(np.arange(0, 61, 10))

#plt.legend()
plt.savefig("引張L.png", format="png", dpi=300,transparent=True)
plt.show()







#応力ひずみ線図 T方向

plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainTave"],df["stress"], label="strainTave",s=1,c="k")
plt.xlabel("StrainT [-]")
plt.ylabel("Stress [MPa]")

#軸の範囲
plt.xlim(-0.005,0)
plt.ylim(0, 60)

#軸の指定
#plt.xticks(np.arange(0, 0.016, 0.005))
plt.yticks(np.arange(0, 61, 10))
#plt.legend()
plt.savefig("引張T.png", format="png", dpi=300,transparent=True)
plt.show()





#ポワソン比

plt.figure(figsize=(7,5),dpi=300)
plt.scatter(df["strainLave"],df["strainTave"], label="poisson",s=1,c="k")

plt.xlabel("StrainL [-]")
plt.ylabel("StrainT [-]")

#軸の範囲
plt.xlim(0,0.020)
plt.ylim(-0.005,0)

#軸の指定
plt.xticks(np.arange(0, 0.021, 0.005))
plt.yticks(np.arange(-0.005, 0.001, 0.001))
#plt.legend()

#見切れるから対策
plt.tight_layout()
plt.savefig("引張P.png", format="png",transparent=True)
plt.show()







