const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");

function getUserMediaSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  alert("해당 브라우저는 지원되지 않습니다.");
}

function enableCam(event) {
  if (!model) {
    return;
  }

  event.target.classList.add("removed");

  const constraints = {
    video: true,
  };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let children = [];

function predictWebcam() {
  model.detect(video).then((predictions) => {
    for (const curChlid of children) {
      liveView.removeChild(curChlid);
    }
    children.splice(0);

    for (const curPredict of predictions) {
      if (curPredict.score > 0.66) {
        const p = document.createElement("p");
        p.innerText =
          curPredict.class +
          " - with " +
          Math.round(parseFloat(curPredict.score) * 100) +
          "% confidence.";
        p.style =
          "margin-left: " +
          curPredict.bbox[0] +
          "px; margin-top: " +
          (curPredict.bbox[1] - 10) +
          "px; width: " +
          curPredict.bbox[2] -
          10 +
          "px; top: 0; left: 0;";
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style = `left: ${curPredict.bbox[0]}px; top: ${curPredict.bbox[1]}px; width: ${curPredict.bbox[2]}px; height: ${curPredict.bbox[3]}px;`;

        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        children.push(highlighter);
        children.push(p);
      }
    }
    window.requestAnimationFrame(predictWebcam);
  });
}

let model = undefined;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  demosSection.classList.remove("invisible");
});
