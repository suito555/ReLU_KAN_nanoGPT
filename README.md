## ReLU_KAN_nanoGPT
[ReLU KAN](https://github.com/quiqi/relu_kan)のソースコードをGPTに組み込んだものです。<br>
<br>
### How to use
```
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char_kan.py
```
train_shakespeare_char_kan.pyのkanの部分を変更することでefficient_KANやMLPが選択できます。
また、実験用にtrain.pyの後半にあるtorch.saveをコメントアウトしているのでモデルを保存したい方は外してください。<br>
```
#print(f"saving checkpoint to {out_dir}")
#torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

KANのパラメーターの感覚がつかめていないため、mlpの設定を固定にしてgrid=5,k=3での結果を表にしておきます。(効果的な設定を知っている方は教えていただけると嬉しいです。)

| モデル名 | パラメータ数 (M) | 処理時間 (ms/step) | train loss | eval loss |
|---|---|---|---|---|
| MLP | 0.80 | 44.43 | 1.7095 | 1.8244 |
| efficient_KAN | 7.87 | 179.77 | 1.5818 | 1.7998 |
| ReLU_KAN | 6.36 | 209.77 | 1.7818 | 1.9501 |
### references
https://github.com/karpathy/nanoGPT
https://github.com/AdityaNG/kan-gpt<br>
https://github.com/AutomaticHourglass/nanoKAN<br>
