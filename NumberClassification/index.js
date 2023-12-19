// MNIST 데이터 세트를 가져옵니다.
import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// 입력 데이터를 변수에 저장합니다.
const INPUTS = TRAINING_DATA.inputs;

// 출력 데이터를 변수에 저장합니다.
const OUTPUTS = TRAINING_DATA.outputs;

// 입력과 출력 데이터를 무작위로 섞습니다.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// 입력 데이터를 텐서로 변환합니다.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// 출력 데이터를 one-hot 인코딩한 후 텐서로 변환합니다.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// 모델을 초기화합니다.
const model = tf.sequential();

// 첫 번째 레이어를 추가합니다. 784개의 입력 노드와 32개의 노드를 가지는 레이어입니다.
model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
// 두 번째 레이어를 추가합니다. 이전 레이어의 출력을 입력으로 받아 16개의 노드를 가집니다.
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
// 마지막 레이어를 추가합니다. 10개의 출력 노드를 가지며, softmax 활성화 함수를 사용합니다.
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

// 모델의 구조를 출력합니다.
model.summary();

// 모델을 훈련시킵니다.
train();

// 모델을 훈련시키는 비동기 함수입니다.
async function train() {
  // 모델을 컴파일합니다. 여기서 손실 함수, 최적화 알고리즘, 평가 지표를 설정합니다.
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 모델을 훈련시킵니다. 여기서 훈련 데이터, 검증 데이터의 비율, 배치 크기, 에폭 수를 설정합니다.
  const results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 50,
    callbacks: { onEpochEnd: logProgress },
  });

  // 훈련이 끝난 후, 텐서를 메모리에서 해제합니다.
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  // 모델을 평가합니다.
  evaluate();
}

// 예측 결과를 출력할 HTML 요소를 가져옵니다.
const PREDICTION_ELEMENT = document.getElementById("prediction");

// 모델을 평가하는 함수입니다.
function evaluate() {
  // 무작위로 입력 데이터 한 개를 선택합니다.
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  // 선택한 데이터를 모델에 입력하고, 결과를 받아옵니다.
  const answer = tf.tidy(() => {
    const newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    const output = model.predict(newInput);
    output.print();
    return output.squeeze().argMax();
  });

  // 결과를 HTML 요소에 표시합니다.
  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

// 이미지를 그릴 캔버스 요소를 가져옵니다.
const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

// 입력 데이터를 이미지로 그리는 함수입니다.
function drawImage(digit) {
  const imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;
    imageData.data[i * 4 + 1] = digit[i] * 255;
    imageData.data[i * 4 + 2] = digit[i] * 255;
    imageData.data[i * 4 + 3] = 255;
  }

  CTX.putImageData(imageData, 0, 0);

  // 일정 시간 후에 다시 모델을 평가합니다.
  setTimeout(evaluate, 2000);
}

// 각 에폭이 끝날 때마다 진행 상황을 로그로 출력하는 함수입니다.
function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}
