makeup_folder="data/makeup/red/test_0/makeup/"
non_makeup_folder="data/makeup/red/test_0/non-makeup/"
test_model=47_1260

for img_makeup in $(ls $makeup_folder)
do
    for img_non_makeup in $(ls $non_makeup_folder)
    do
        python test_single.py --img_src=$non_makeup_folder$img_non_makeup --img_ref=$makeup_folder$img_makeup --test_model=$test_model
    done
done

# makeup_folder="data/makeup/red/detect_align/outputs/red/makeup/"
# non_makeup_folder="data/makeup/red/detect_align/outputs/red/non-makeup/"
# test_model=112_1260
# 
# for img_makeup in $(ls $makeup_folder)
# do
#     for img_non_makeup in $(ls $non_makeup_folder)
#     do
#         python test_single.py --img_src=$non_makeup_folder$img_non_makeup --img_ref=$makeup_folder$img_makeup --test_model=$test_model
#     done
# done


# img_src=data/makeup/red/detect_align/outputs/red/non-makeup/ef955aa9-fd07-3f70-af16-617110e19429_1.png
# img_ref=data/makeup/red/detect_align/outputs/red/makeup/5a597630-c15f-5620-84e2-5be064e52f52_0.png
# # img_ref=data/odds/detected/odd1_0.png
# test_model=112_1260
# 
# python test_single.py --img_src=$img_src --img_ref=$img_ref --test_model=$test_model
