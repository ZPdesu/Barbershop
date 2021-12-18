import gdown
import os


weight_dic = {'afhqwild.pt': 'https://drive.google.com/file/d/14OnzO4QWaAytKXVqcfWo_o2MzoR4ygnr/view?usp=sharing',
                'afhqdog.pt': 'https://drive.google.com/file/d/16v6jPtKVlvq8rg2Sdi3-R9qZEVDgvvEA/view?usp=sharing',
                'afhqcat.pt': 'https://drive.google.com/file/d/1HXLER5R3EMI8DSYDBZafoqpX4EtyOf2R/view?usp=sharing',
                'ffhq.pt': 'https://drive.google.com/file/d/1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS/view?usp=sharing',
                'metfaces.pt': 'https://drive.google.com/file/d/16wM2PwVWzaMsRgPExvRGsq6BWw_muKbf/view?usp=sharing',
                'seg.pth': 'https://drive.google.com/file/d/1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62/view?usp=sharing'

}


def download_weight(weight_path):
    gdown.download(weight_dic[os.path.basename(weight_path)],
                   output=weight_path, fuzzy=True)
