# SUPER TML

![]("./imgs/tabular_to_img.PNG")

"Super Characters: A Conversion from Sentiment Classification to Image Classification"를 모티브로한 논문이다. 
해당 논문은 글자를 이미지를 이용해서 분류하는 방법론을 정형데이터도 이미지로 만들어서 적용하였다. 
실제로 이러한 방법론이 됬으면 좋겠다고 생각은 하고 있었으나, 필자는 그 숫자값을 3차원으로 표현해서 시도를 했었고, 이 저자는 숫자들을 이미지화해서 하니 잘된 것을 보였다. 그래서 정형 데이터에 이러한 시도를 한 것이 너무나 반갑다.

## 인상깊은 점

제안된 SuperTML 방법은 숫자 값으로 전처리 할 필요없이 표 형식 데이터의 범주형 데이터와 결측값을 자동으로 처리

위의 문장이 아주 충격적이라서 더 관심을 가지게 되었다.
정형 데이터를 분석하기 위해서는 feature engineering이 너무나도 중요하고 많은 시간을 할애하는데, 저 논문에서는 마치 이미지에서 전처리 조금하듯이 거의 전처리를 디테일하게 할 필요없이 넣기만 하면 잘된다라고 주장하는 것이다. 

## 방법

저자는 분류 문제를 풀기 위해서 1차원으로 2차원으로 투영하는 방식을 선택했다. 그리고 여기서 그냥 CNN을 구성하는게 아니라 이미 학습된 resnet이나 denset같은 것도 사용할 수 있다. 즉 이말은 pretrained model을 사용하여, 보다 빠른 학습을 가능하게 할 수 있다는 것으로 들려서, 더 특별하다고 느껴졌다.

그리고 저자는 데이터를 이미지화하는데서 크게 2가지를 제안하고 있다.
1. SuperTML_VF(Variant Font)
2. SuperTML_EF(Equal Font)

![]("./imgs/data2img.PNG")
![]("./imgs/data2img_2.PNG")

## 실험 결과

![]("./imgs/EXP.PNG")

논문

![SuperTML: Two-Dimensional Word Embedding for the Precognition on Structured Tabular Data](https://arxiv.org/abs/1903.06246)
![A novel method for classification of tabular data using convolutional neural networks](https://www.biorxiv.org/content/10.1101/2020.05.02.074203v1.full)