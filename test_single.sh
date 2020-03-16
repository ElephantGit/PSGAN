# img_src=data/makeup/makeup_final/images/non-makeup/vSYYZ536.png
img_src=data/makeup/red/test_0/non-makeup/1d0b45d9-7b36-32d4-b8c8-622bd743b482_2.png
img_ref=data/makeup/red/test_0/makeup/3a695f04-ca5e-358e-98d9-45a102f31c99.png
test_model=4_1260
python test_single.py --img_src=$img_src --img_ref=$img_ref --test_model=$test_model

# # img_src=data/makeup/makeup_final/images/non-makeup/vSYYZ536.png
# img_src=data/makeup/red/detect_align/outputs/red/non-makeup/ef955aa9-fd07-3f70-af16-617110e19429_1.png
# img_ref=data/makeup/red/detect_align/outputs/red/makeup/5a597630-c15f-5620-84e2-5be064e52f52_0.png
# # img_ref=data/odds/detected/odd1_0.png
# test_model=112_1260
# 
# python test_single.py --img_src=$img_src --img_ref=$img_ref --test_model=$test_model
