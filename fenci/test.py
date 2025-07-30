import pkuseg
import pandas as pd

# 对input.txt的文件分词输出到output.txt中
# 开20个进程
if __name__=='__main__':
    # seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
    input_files = ['input_train.txt', 'input_test.txt']
    output_files = ['output_train.txt', 'output_test.txt']

    # for in_file, out_file in zip(input_files, output_files):
    #     pkuseg.test(in_file, out_file,user_dict="./通信工程.txt",nthread=5)
    #     print(f"处理完成：{in_file} -> {out_file}")


    words = []
    for file in output_files:
        with open(file, 'r', encoding='utf-8') as f:
            # 逐行读取，strip()去掉每行两端的空白（如换行符、空格），过滤空行
            question_list = [line.strip() for line in f if line.strip()]
            for line in question_list:
                line1=line.split()
                words.extend(line1)

    

    # 数据准备
    corpus = pd.DataFrame(words, columns=['word'])
    corpus['cnt'] = 1
    g = corpus.groupby(['word']).agg({'cnt': 'count'}).sort_values('cnt', ascending=False)

    # 重置索引并重命名列（创建友好的DataFrame）
    friendly_df = g.reset_index()
    friendly_df.columns = ['Word', 'Count']  # 更友好的列名

    # 输出到TXT文件（自定义格式）
    with open('word_frequencies.txt', 'w', encoding='utf-8') as f:
        # 写入标题和表头
        f.write("Word Frequency Report\n")
        f.write("=====================\n")
        f.write(f"Total unique words: {len(friendly_df)}\n")
        f.write("---------------------\n")
        
        # 创建格式化的表格
        # 计算列宽（动态适应数据长度）
        word_width = max(friendly_df['Word'].apply(len).max(), len('Word')) + 2
        count_width = max(len(str(friendly_df['Count'].max())), len('Count')) + 2
        
        # 写入列标题
        f.write(f"{'Word':<{word_width}}{'Count':>{count_width}}\n")
        f.write('-' * (word_width + count_width) + '\n')
        
        # 逐行写入数据
        for _, row in friendly_df.iterrows():
            f.write(f"{row['Word']:<{word_width}}{row['Count']:>{count_width}}\n")
        
        # 添加汇总信息
        f.write("\nReport generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

