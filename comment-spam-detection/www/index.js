const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const COMMENTS_LIST = document.getElementById("commentsList");
const PROCESSING_CLASS = "processing";

function handleCommentPost() {
  if (!POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
    POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
    COMMENT_TEXT.classList.add(PROCESSING_CLASS);
    const currentComment = COMMENT_TEXT.innerText;
    console.log(currentComment);
  }
}

POST_COMMENT_BTN.addEventListener("click", handleCommentPost);

const MODEL_JSON_URL =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/model.json";
const SPAM_THRESHOLD = 0.75;
let model = undefined;

async function loadAndPredict(inputTensor) {
  if (model === undefined) model = await tf.loadLayersModel(MODEL_JSON_URL);

  const results = await model.predict(inputTensor);

  results.print();
}

loadAndPredict(
  tf.tensor([[1, 3, 12, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
);
