import yaml


with open('src/deblurganv2/config/config.yaml', encoding='utf-8') as cfg:
    config: dict = yaml.load(cfg, Loader=yaml.FullLoader)
