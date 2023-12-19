import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.2,
    batchSize: 512,
    epochs: 50,
    callbacks: { onEpochEnd: logProgress },
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate();
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  const answer = tf.tidy(() => {
    const newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    const output = model.predict(newInput);
    output.print();
    return output.squeeze().argMax();
  });

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
