
## svg库部分

- dataset_svg
- dataset_graph
- dataset_img


## python文件部分

- main.py
  - 处理库的文件
- read.py：读取svg的文件
- dataset_build.py：读取dataset_img和dataset_graph里的中间形式文件，并转换为嵌入向量库
- predictor.py：CNN模型预测层
- encoder.py：CNN模型编码器

## 输出部分

- input:里面装的是输入的svg
- middle:中间处理得到的png和graph文件
- output:最终的输出
- cnn.tsv:视觉信息嵌入向量库
- gnn.tsv:结构信息嵌入向量库
- storage.tsv:嵌入向量库