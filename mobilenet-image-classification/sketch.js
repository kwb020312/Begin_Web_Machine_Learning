let mobilenet;

let penguin;

function modelReady() {
  console.log("Model is Ready!!!");
  mobilenet.predict(penguin, gotResults);
}

function gotResults(error, results) {
  if (error) console.error(error);
  else {
    console.log(results);
    const label = results[0].label;
    const prob = results[0].confidence;
    fill(0);
    textSize(64);
    text(label, 10, height - 100);
    createP(label);
    createP(prob);
  }
}

function imageReady() {
  image(penguin, 0, 0, width, height);
}

function setup() {
  createCanvas(640, 480);
  background(0);
  penguin = createImg("images/penguin.jpg", imageReady);
  penguin.hide();
  background(0);
  mobilenet = ml5.imageClassifier("MobileNet", modelReady);
}

// function draw() {
//   background(220);
// }
