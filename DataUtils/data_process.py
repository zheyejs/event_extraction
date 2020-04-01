# !/usr/bin/env python
# -*- coding: utf8 -*-
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#

"""
@author: jiangshuai(jiangshuai02@baidu.com)
@file: data_process.py
@time: 2020/3/31 21:32
@desc: 事件数据预处理
"""
import json


class DataProccess(object):

    def __init__(self, schema_file):
        """
        初始化
        """
        self.bio = dict()
        self.bio_reverse = dict()
        # 初始化bio
        self.set_data_bio(schema_file)

    def set_data_bio(self, file_name):
        """
        基于原始事件scheam获取标注的boi
        :param file_name:
        :return:
        """
        index = 0
        with open(file_name, 'r') as f:
            for line in f:
                l_j = json.loads(line.strip())
                for item in l_j['role_list']:
                    role = item['role']
                    if role not in self.bio:
                        self.bio[role] = str(index)
                        self.bio_reverse[str(index)] = role
                        index += 1

    def _get_stander_sen(self, one_json_line):
        """
        :param one_json_line:
        '{"text": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
         "id": "409389c96efe78d6af1c86e0450fd2d7", "event_list":
          [{"event_type": "组织关系-裁员", "trigger": "裁员",
           "trigger_start_index": 2, "arguments":
            [{"argument_start_index": 0, "role": "裁员方",
             "argument": "雀巢", "alias": []}, {"argument_start_index":
              4, "role": "裁员人数", "argument": "4000人", "alias": []}],
               "class": "组织关系"}]}'
        :return:
        """
        l_j = json.loads(one_json_line)
        # 解析出标记点
        index_role_dict, index_argument_dict = dict(), dict()
        total_index = list()
        for item in l_j['event_list'][0]['arguments']:
            a_s_i, role = item['argument_start_index'], item['role']
            index_role_dict[str(a_s_i)] = 'B_' + self.bio[role]
            for i, ii in enumerate(range(len(item['argument']) - 1), start=1):
                index_role_dict[str(a_s_i + i)] = 'I_' + self.bio[role]
        # bio list
        bio_list = list()
        for index, word in enumerate(l_j['text']):
            if str(index) in index_role_dict:
                bio_list.append(word + ' ' + index_role_dict[str(index)])
            else:
                bio_list.append(word + ' ' + 'O')
        l_j['bio_list'] = bio_list
        return l_j

    def get_stander_data(self, file_name, out_file_name):
        """
        基于json数据标准化用于训练的数据
        :param file_name:
        :param out_file_name:
        :return:
        """
        with open(out_file_name, 'w') as f:
            pass
        with open(out_file_name, 'w') as f:
            with open(file_name, 'r') as f1:
                for line in f1:
                    stander_data = self._get_stander_sen(line.strip())
                    f.write('\n'.join(stander_data['bio_list']))
                    f.write('\n\n')



if __name__ == '__main__':
    """
    """
    dp = DataProccess('data/event_schema.json')
    print(dp.bio)
    print(dp._get_stander_sen('{"text": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！", "id": "409389c96efe78d6af1c86e0450fd2d7", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 2, "arguments": [{"argument_start_index": 0, "role": "裁员方", "argument": "雀巢", "alias": []}, {"argument_start_index": 4, "role": "裁员人数", "argument": "4000人", "alias": []}], "class": "组织关系"}]}'))
    dp.get_stander_data('data/train.json', 'data/train.data')
