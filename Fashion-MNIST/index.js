import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js";

const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return NORMALIZED_VALUES;
  });

  return result;
}

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
console.log({ INPUTS });
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const LOOKUP = [
  "T-shirt",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];

const model = tf.sequential();

model.add(
  tf.layers.conv2d({
    // 28 x 28 이미지며 흑색이므로 색상 채널은 1
    inputShape: [28, 28, 1],
    // 입력 이미지가 처리될 3x3 정사각형 필터 16개
    filters: 16,
    kernelSize: 3,
    strides: 1,
    // 3x3 정사각형 필터가 배열 크기를 초과할 경우 padding을 씌워줌
    padding: "same",
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

model.add(
  tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

// 모델의 구조를 출력합니다.
model.summary();

train();

async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);
  const results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.15,
    epochs: 30,
    batchSize: 256,
    callbacks: { onEpochEnd: logProgress },
  });
  RESHAPED_INPUTS.dispose();
  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  evaluate();
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  const answer = tf.tidy(() => {
    const newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();
    const output = model.predict(newInput.reshape([1, 28, 28, 1]));
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = LOOKUP[index];
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

function drawImage(digit) {
  const imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255;
    imageData.data[i * 4 + 1] = digit[i] * 255;
    imageData.data[i * 4 + 2] = digit[i] * 255;
    imageData.data[i * 4 + 3] = 255;
  }

  CTX.putImageData(imageData, 0, 0);

  setTimeout(evaluate, 2000);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

let interval = 2000;
const RANGER = document.getElementById("ranger");
const DOM_SPEED = document.getElementById("domSpeed");

RANGER.addEventListener("input", (e) => {
  interval = this.value;
  DOM_SPEED.innerText = `Classification 시간을 조절해보세요! 현재는 ${interval}ms 입니다.`;
});
