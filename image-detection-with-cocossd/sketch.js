// Image Detection

// let img;
// let detector;

// function preload() {
//   img = loadImage("./images/puppy-cat.jpg");
//   detector = ml5.objectDetector("cocossd");
// }

// function setup() {
//   createCanvas(innerWidth, innerHeight);
//   image(img, 0, 0, width, height);
//   console.log(detector);
//   detector.detect(img, (error, results) => {
//     console.log(results);
//     for (const object of results) {
//       stroke(0, 255, 0);
//       strokeWeight(4);
//       noFill();
//       rect(object.x, object.y, object.width, object.height);
//       noStroke();
//       fill(0);
//       textSize(24);
//       text(object.label, object.x + 10, object.y + 24);
//     }
//   });
// }

// Webcam Detection
let video;
let detector;
let detections = [];

function preload() {
  detector = ml5.objectDetector("cocossd");
}

function setup() {
  createCanvas(innerWidth, innerHeight);
  video = createCapture(VIDEO, gotDetections);
  video.size(640, 480);
  video.hide();
}

function gotDetections(error, results) {
  if (error) {
    console.error(error);
  }
  detections = results;
  detector.detect(video, gotDetections);
}

function draw() {
  image(video, 0, 0);
  for (const object of detections) {
    stroke(0, 255, 0);
    strokeWeight(4);
    noFill();
    rect(object.x, object.y, object.width, object.height);
    noStroke();
    fill(0);
    textSize(24);
    text(object.label, object.x + 10, object.y + 24);
  }
}
