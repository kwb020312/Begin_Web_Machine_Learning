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
