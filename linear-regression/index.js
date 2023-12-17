import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

// 입력값과 출력 값의 인덱스는 그대로 매치시킨 채 요소를 무작위 정렬
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// 입력은 2차원, 출력은 1차원
const INPUTS_TENSOR = tf.tensor2d(INPUTS);
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
