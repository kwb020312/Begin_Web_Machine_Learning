const POST_COMMENT_BTN = document.getElementById("post");
const COMMENT_TEXT = document.getElementById("comment");
const COMMENTS_LIST = document.getElementById("commentsList");
const PROCESSING_CLASS = "processing";

function handleCommentPost() {
  if (!POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
    POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
    COMMENT_TEXT.classList.add(PROCESSING_CLASS);
    const currentComment = COMMENT_TEXT.innerText;

    const lowercaseSentenceArray = currentComment
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(" ");

    const li = document.createElement("li");
    const p = document.createElement("p");
    p.innerText = COMMENT_TEXT.innerText;
    const spanName = document.createElement("span");
    spanName.setAttribute("class", "username");
    spanName.innerText = "AnonyMous User";
    const spanDate = document.createElement("span");
    const curDate = new Date();
    spanDate.innerText = curDate.toLocaleString();
    li.appendChild(spanName);
    li.appendChild(spanDate);
    li.appendChild(p);
    COMMENTS_LIST.prepend(li);
    COMMENT_TEXT.innerText = "";

    loadAndPredict(tokenize(lowercaseSentenceArray), li).then(() => {
      POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
      COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
    });
  }
}

POST_COMMENT_BTN.addEventListener("click", handleCommentPost);

const MODEL_JSON_URL =
  "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/model.json";
const SPAM_THRESHOLD = 0.75;
let model = undefined;

async function loadAndPredict(inputTensor, domComment) {
  if (model === undefined) model = await tf.loadLayersModel(MODEL_JSON_URL);

  const results = await model.predict(inputTensor);

  results.print();

  const dataArray = results.dataSync();
  if (dataArray[1] > SPAM_THRESHOLD) domComment.classList.add("spam");
  else {
    socket.emit("comment", {
      username: "AnonyMous",
      timestamp: domComment?.querySelectorAll("span")[1].innerText,
      comment: domComment?.querySelectorAll("p")[0].innerText,
    });
  }
}

// loadAndPredict(
//   tf.tensor([[1, 3, 12, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
// );

import * as DICTIONARY from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/spam/dictionary.js";

const ENCODING_LENGTH = 20;

function tokenize(wordArray) {
  const returnArray = [DICTIONARY.START];

  for (let i = 0; i < wordArray.length; i++) {
    const encoding = DICTIONARY.LOOKUP[wordArray[i]];
    returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  }

  while (returnArray.length < ENCODING_LENGTH) {
    returnArray.push(DICTIONARY.PAD);
  }

  console.log([returnArray]);

  return tf.tensor2d([returnArray]);
}

const socket = io.connect();

function handleRemoteComments(data) {
  const li = document.createElement("li");
  const p = document.createElement("p");
  p.innerText = data.comment;

  const spanName = document.createElement("span");
  spanName.setAttribute("class", "username");
  spanName.innerText = data.username;

  const spanDate = document.createElement("span");
  spanDate.setAttribute("class", "timestamp");
  spanDate.innerText = data.timestamp;

  li.appendChild(spanName);
  li.appendChild(spanDate);
  li.appendChild(p);

  COMMENTS_LIST.prepend(li);
}

socket.on("remoteComment", handleRemoteComments);
