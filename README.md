# 😀 Why AI?

AI는 현대 사회에서 가장 각광받는 기술이며, 프론트엔드 개발자일지라도 기본적인 이해와 사고 방식을 더불어 사용할 수 있어야 한다는 생각이 들어 제작하게 되었음
웹 개발자도 더이상은 AI의 영역을 벗어날 수 없으며, 복잡한 계산과 미래지향적인 웹 개발을 진행하기 위해 AI에 대한 이해는 반드시 이루어져야 한다.
예를들어 작성자의 경우, Three.JS로 생성된 방 내부에서 사용자 캠의 시각에 따라 카메라 시점을 움직여야 할 필요가 있었기에 AI를 배우게 되었다.

## 😂Why JavaScript?

google에서 개발한 라이브러리로, 본래 Python을 통해 활용하기위해 개발하였으나, JavaScript 사용자가 많음을 인지하고 웹 개발을 통해서도 다양하게 사용될 수 있도록 제작되었음

1. 다양한 분야에서 하나의 언어로 실행이 가능하다

![image](https://github.com/kwb020312/Begin_Web_Merchine_Learning/assets/46777310/8dd99cd4-8ba4-4b65-a593-4b0809923528)

2. 사용 가능한 사용자 자원이 매우 많다

![image](https://github.com/kwb020312/Begin_Web_Merchine_Learning/assets/46777310/8f88a1ab-4215-46e5-b98d-7cf5aaf35589)

3. NodeJS로 서버까지 구축될 경우 Python을 사용한 모델보다 성능이 우수하다

![image](https://github.com/kwb020312/Begin_Web_Merchine_Learning/assets/46777310/c1bfe65a-c672-43b2-9b71-d43727a1c0ea)
![image](https://github.com/kwb020312/Begin_Web_Merchine_Learning/assets/46777310/3d9d017b-e8e9-4b3a-91f6-e435cb270ed6)

---

## 😀Base Of Tensorflow

데이터를 다차원의 형태로 표현해야할 때, VanilaJS를 사용한다면 다차원 배열 계산 수행시간이 복잡해질수록 너무 오래 걸리기 때문에 해당 라이브러리의 자료형으로 표현하고 계산해야한다.

```javascript
tf.scalar(1) // 정수 1(스칼라)
tf.tensor1d([1, 2, 3]) // 1차원 배열
tf.tensor2d(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
) // 2차원 배열
...
tf.tensorxd(...) // N차원 배열 (최대 6차원)
```

---

## 😁Load Model

생성된 모델을 불러와 입 출력을 확인해보자

```javascript
// 평당 주택 가격을 예상하는 예시 모델
const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";
let model = undefined;

async function loadModel() {
  // 모델은 레이어 형식과 그래프 형식이 있음
  // 레이어: 사람이 이해하기 좋으나 속도 느림
  // 그래프: 사람이 이해하기 어려우나 속도 빠름
  model = await tf.loadLayersModel(MODEL_PATH);
  // summary를 통해 모델의 입출력 구조를 확인
  // input 형식이 [null, 1] 이며, 이는 N개의 입력을 한번에 받을 수 있고 입력 배열요소 수는 1개여야 함을 의미
  model.summary();

  // 단일 입력
  const input = tf.tensor2d([[870]]);

  // 배치 입력
  const inputBatch = tf.tensor2d([[500], [1100], [970]]);

  // 추론
  const result = model.predict(input);
  const resultBatch = model.predict(inputBatch);

  // 결과
  result.print();
  resultBatch.print();

  // 폐기
  input.dispose();
  inputBatch.dispose();
  result.dispose();
  resultBatch.dispose();
  model.dispose();
}

loadModel();
```

---

## 💾TFHub

Tensorflow에서 활용할 수 있는 [오픈소스 모델을 공유하는 사이트](https://www.kaggle.com/models?tfhub-redirect=true)이며, 각 모델은 사용법을 명시해주기 때문에 사용 전 참고하는것이 좋다.

## 💤Image Classification

`ml5.js`를 통해 `MobileNet`에 저장된 1000여가지의 Class를 분류해주는 Image Classification이 가능하다.

이미지 및 동영상에 해당 모델을 불러온 후 내장된 예측 메서드를 사용하여 결괏값을 받을 수 있다.

![image](https://github.com/kwb020312/Begin_Web_Merchine_Learning/assets/46777310/c51730a5-72f9-4d14-8ead-7ddc67b8d107)

---

## 🚼Object Detection

`ml5.js`의 `cocossd`를 활용한 Object Detection이 가능하다.

![image](https://github.com/kwb020312/Begin_Web_Machine_Learning/assets/46777310/131adc56-d048-44ff-acd5-5b7871cafa02)

---

## 🏃‍♂️Pose Estimation

```javascript
const MODEL_PATH = `https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4`;
const EXAMPLE_IMG = document.getElementById("exampleImg");
let movenet = undefined;

async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

  //   const exampleInputTensor = tf.zeros([1, 192, 192, 3], "int32");
  // 이미지 태그의 픽셀을 텐서로 변환
  const imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  console.log(imageTensor.shape);

  // 자르기 시작할 위치
  const cropStartPoint = [15, 170, 0];
  // 자를 이미지 크기
  const cropSize = [345, 345, 3];
  // 텐서 자르기
  const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

  // 자른 텐서를 기반한 모델 입력 사이즈 맞추기 192
  const resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();

  const tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  const arrayOutput = await tensorOutput.array();

  console.log(arrayOutput);
}

loadAndRunModel();
```

---
