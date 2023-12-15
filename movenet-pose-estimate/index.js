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
