import pkuseg
import pandas as pd

# 对input.txt的文件分词输出到output.txt中
# 开20个进程
if __name__=='__main__':
    # seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
    input_files = ['三国演义.txt', '西游记.txt']
    output_files = ['三国演义_fenci.txt', '西游记_fenci.txt']

    for in_file, out_file in zip(input_files, output_files):
        pkuseg.test(in_file, out_file,nthread=5)
        print(f"处理完成：{in_file} -> {out_file}")

