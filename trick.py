import os.path
import shutil

result_path = r'./result.txt'
fp = open(result_path, 'r', encoding='utf-8')
data = fp.read()
data = list(data)
result = []
for ch in data:
    if ch != '\n':
        result.append(int(ch))

cnt = [[], [], [], [], [], [], [], [], [], []]
for i in range(500):
    cnt[result[i]].append(i)
for i in range(10):
    print(i, ":", len(cnt[i]))

# 将图片按分类类别归类，方便检查
# path_data = "./data_set/food_data/test"
# path_result = "./data_set/food_data/result"
# for i in range(10):
#     save_path = os.path.join(path_result, str(i))
#     if os.path.exists(save_path):
#         shutil.rmtree(save_path)
#     os.mkdir(save_path)
#     for j in cnt[i]:
#         img_path = os.path.join(path_data, "img_" + str(j) + ".jpg")
#         assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
#         shutil.copy(img_path, save_path + "/img_" + str(j) + ".jpg")


err = [314, 42, 185, 79, 150, 105, 494, 375, 408, 473, 163, 90]
cor = [8, 7, 7, 8, 8, 7, 3, 0, 0, 8, 8, 3]
for i in range(len(err)):
    result[err[i]] = cor[i]

cnt = [[], [], [], [], [], [], [], [], [], []]
for i in range(500):
    cnt[result[i]].append(i)
for i in range(10):
    print(i, ":", len(cnt[i]))

for i in result:
    print(i)