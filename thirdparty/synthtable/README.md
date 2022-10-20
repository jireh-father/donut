# SynthDoG 🐶: Synthetic Document Generator

SynthDoG is synthetic document generator for visual document understanding (VDU).

![image](../misc/sample_synthdog.png)

## Summary
```text
배경
문서
 페이퍼
 컨텐츠(텍스트 layers)
  레이아웃
   텍스트박스(with font)

컨텐츠 + 페이퍼

[페이퍼+컨텐츠] + 배경

배경
문서
 테이블
   페이퍼(배경)
   컨텐츠(table html) : 공통 속성, 폰트, 사이즈, 등등
    header : 개별 속성, 폰트, 사이즈, 등등
    rows : 개별 속성, 폰트, 사이즈, 등등
    cells : 개별 속성, 폰트, 사이즈, 등등
     text : 개별 속성, 폰트, 사이즈, 등등
```


## Prerequisites

- python>=3.6
- [synthtiger](https://github.com/clovaai/synthtiger) (`pip install synthtiger`)

## Usage

```bash
# Set environment variable (for macOS)
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

synthtiger -o ./outputs/SynthDoG_en -c 50 -w 4 -v template.py SynthDoG config_en.yaml

{'config': 'config_en.yaml',
 'count': 50,
 'name': 'SynthDoG',
 'output': './outputs/SynthDoG_en',
 'script': 'template.py',
 'verbose': True,
 'worker': 4}
{'aspect_ratio': [1, 2],
     .
     .
 'quality': [50, 95],
 'short_size': [720, 1024]}
Generated 1 data
Generated 2 data
Generated 3 data
     .
     .
Generated 49 data
Generated 50 data
46.32 seconds elapsed
```

Some important arguments:

- `-o` : directory path to save data.
- `-c` : number of data to generate.
- `-w` : number of workers.
- `-v` : print error messages.

To generate ECJK samples:
```bash
# english
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_en.yaml

# chinese
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_zh.yaml

# japanese
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_ja.yaml

# korean
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_ko.yaml
```
