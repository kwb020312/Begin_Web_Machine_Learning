// 평당 주택 가격을 예상하는 예시 모델
const MODEL_PATH =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json";
let model = undefined;

async function loadModel() {
  // 모델은 레이어 형식과 그래프 형식이 있음
  // 레이어: 사람이 이해하기 좋으나 속도 느림
  // 그래프: 사람이 이해하기 어려우나 속도 빠름
  model = await tf.loadLayersModel(MODEL_PATH);
  // input 형식이 [null, 1] 이며, 이는 N개의 입력을 한번에 받을 수 있고 입력 형식은 1개여야 함을 의미
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
