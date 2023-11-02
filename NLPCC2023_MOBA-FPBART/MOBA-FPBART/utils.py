import json
import pandas as pd

class UtilsClass:
    def open_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data

    def open_txt(self, file_path):
        with open(file_path, "r", encoding='utf-8') as file:
            content_list = file.readlines()
        content_list = [line.strip() for line in content_list]
        return content_list

    def open_csv(self, file_path):
        data = pd.read_csv(file_path, delimiter='\t', header=0, encoding='utf-8')
        return data

    def save_df2txt(self, data, file_path):
        data.to_csv(file_path, index=False, sep='\t') 

    def save_list2txt(self, my_list, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in my_list:
                file.write(str(item) + '\n')

    def save_dict2json(self, my_dict, file_path):
        final = json.dumps(my_dict, indent=4, ensure_ascii=False)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final)

    def save_list2json(self, my_list, file_path):
        json_string = json.dumps(my_list, indent=4, ensure_ascii=False)
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(json_string)

    def open_src(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.rstrip() for line in f.readlines()]
        return data