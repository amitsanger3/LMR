from sklearn import preprocessing
import torch


class OpenDataset(object):

    def __init__(self, d_word_index, embed, process_dict, text_len=64, tag_name='O') -> None:
        self.d_word_index = d_word_index
        self.embed = embed
        self.process_dict = process_dict
        self.text_len = text_len
        self.tag_name = tag_name

    def __len__(self):
        return len(self.process_dict[list(self.process_dict.keys())[0]])

    def __getitem__(self, item):

        def mark_tag(tag):
            if tag == self.tag_name:
                return 0
            else:
                return 1

        text = self.process_dict['raw_words'][item]
        # pos = self.pos[item]
        tags = self.process_dict['raw_targets'][item]

        ids = [self.embed[self.d_word_index['__PADDING__']]]
        target_pos = []
        target_tag = []

        for i in range(self.text_len - 1):

            if i < len(text):
                if text[i] in self.d_word_index.keys():
                    ids.append(self.embed[self.d_word_index[text[i]]])
                else:
                    ids.append(self.embed[self.d_word_index['__UNK__']])
            else:
                ids.append(self.embed[self.d_word_index['__PADDING__']])

        # print("i m here >>")
        target_tag = list(map(mark_tag, tags))
        enc_tag = preprocessing.LabelEncoder()
        enc_tag.fit_transform(target_tag)

        # ids = ids[:config['MAX_LEN'] - 2]
        # target_pos = target_pos[:config['MAX_LEN'] - 2]
        target_tag = target_tag[:self.text_len - 2]

        # ids = [101] + ids + [102]
        # target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        # mask = [1] * len(ids)
        # token_type_ids = [0] * len(ids)

        padding_len = self.text_len - len(target_tag)

        # ids = ids + ([0] * padding_len)
        # mask = mask + ([0] * padding_len)
        # token_type_ids = token_type_ids + ([0] * padding_len)
        # target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.stack(ids).to(torch.float),
            # "mask": torch.tensor(mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            # "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.float),
        }

