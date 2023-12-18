import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

const INPUTS = Array.from({ length: 20 }, (_, i) => i + 1);
const OUTPUTS = Array.from({ length: 20 }, (_, i) => (i + 1) ** 2);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// 정규화된 Tensor를 반환하는 함수
function normalize(tensor, min, max) {
  // tf.tidy()는 메모리 관리를 위해 사용되며, 이 안에서 생성된 텐서들은 이 블록이 끝난 후에 자동으로 해제됩니다.
  const result = tf.tidy(() => {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    // 텐서에서 최소값을 뺀 결과를 TENSOR_SUBTRACT_MIN_VALUE에 저장
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // 최댓값 - 최솟값을 통해 0~1 범위를 결정하는 '분모' 생성
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    // 0~1사잇값 반환
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log("Normalized Values: ");
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values: ");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values: ");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

model.summary();

const LEARNING_RATE = 0.01;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

train();

async function train() {
  // 최적화 및 손실함수 입력
  model.compile({
    optimizer: OPTIMIZER,
    loss: "meanSquaredError",
  });

  // 학습진행 정규화된 값과 출력 텐서를 전달
  const results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress },
      shuffle: true,
      batchSize: 2,
      epochs: 200,
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log("Average error loss: " + Math.sqrt(results.history.loss.at(-1)));

  evaluate();
}

function evaluate() {
  tf.tidy(() => {
    const newInput = normalize(
      tf.tensor1d([7]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    const output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, Math.sqrt(logs.loss));
  if (epoch == 70) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}
