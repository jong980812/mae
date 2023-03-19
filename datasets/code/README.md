
# ASD 선별 그림 데이터 분석을 위한 특징추출 및 분류 기술연구 

This is the official code for ASD 선별 그림 데이터 분석을 위한 특징추출 및 분류 기술연구 

## Dataset
- person drawing, 인물화 그림
- free drawing, 자유화 그림

- Half_crop - 인물화 그림 데이터에 half crop을 적용한 데이터셋
- Hand_crop - 인물화 그림 데이터에 hand crop을 적용한 데이터셋

## ASD 선별 그림 데이터 분석을 위한 특징추출 및 분류 기술연구 결과 

### 지유화
| Method       |       Result     |  Improvment     |
|-------------|------------|--------|
| ResNet |     96.35(±4.82)      |  + 0.0  |
| ResNet + Transfer Learning |    98.14 (±02.50)      |  + 1.79  |

### 인물화
| Method       |       Result     |  Improvment     |
|-------------|------------|--------|
| ResNet |     78.94 (±08.92)      |  + 0.0  |
| ResNet + Data Augmentation (Crop) |     91.11 (±2.79)      |  + 12.17  |
| ResNet + Data Augmentation (ALL) |     87.18 (±08.66)      |  + 8.24  |
| ResNet + Data Augmentation (Binary) |     84.34 (±08.68)      |  + 16.13  |
| ResNet + DA(Binary+crop) |     95.07 (±03.60)     |  + 0.0  |
| ResNet + DA(Binary+crop) + Transfer learning |     94.48 (±02.93)      |  + 15.54  |


## Quick start
- Execute the following code
```ruby
bash train_resnet_person.sh
```
