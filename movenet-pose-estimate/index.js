const MODEL_PATH = `https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4`;
const EXAMPLE_IMG = document.getElementById("exampleImg");
let movenet = undefined;

async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

  const exampleInputTensor = tf.zeros([1, 192, 192, 3], "int32");
  const imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
  console.log(imageTensor.shape);

  const cropStartPoint = [15, 170, 0];
  const cropSize = [345, 345, 3];
  const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

  const resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();

  const tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  const arrayOutput = await tensorOutput.array();

  console.log(arrayOutput);
}

loadAndRunModel();
