const WIDTH = 700;
const HEIGHT = 700;
const STROKE_WEIGHT = 3;
const CROP_PADDING = (REPOS_PADDING = 2);

let model;
let pieChart;
let clicked = false;
let mousePosition = [];
let score = 0;
let isBlocking = false;

let currentRound = 1;
let maxRounds = 6;
let roundResults = [];

// Coordinates of the current drawn stroke [[x1, x2, ..., xn], [y1, y2, ..., yn]]
let strokePixels = [[], []];

// Coordinates of all canvas strokes [[[x1, x2, ..., xn], [y1, y2, ..., yn]], [[x1, x2, ..., xn], [y1, y2, ..., yn]], ...]
let imageStrokes = [];

let intervalId;
let timerCount = 20;

const startTimer = (seconds) => {
  const timerDiv = document.getElementById("timer");
  timerDiv.innerText = `${seconds} seconds left`;

  intervalId = setInterval(() => {
    seconds--;
    if (seconds < 0) {
      clearInterval(intervalId);
      timerDiv.innerText = "Time's up!";
      blockDrawingAndPrediction();
    } else {
      timerDiv.innerText = `${seconds} seconds left`;
    }
  }, 1000);
};

const resetApplication = () => {
  if (currentRound > maxRounds) {
    endGame();
    return;
  }

  clearInterval(intervalId); // Clear the existing interval
  clearCanvas();
  setRandomLabel();
  startTimer(timerCount);
  isBlocking = false;

  window.addEventListener("mousedown", mouseDown);
  window.addEventListener("mousemove", mouseMoved);
  document.getElementById("clearButton").disabled = false;

  // Hide the reload/next button
  document.getElementById("reloadButton").style.display = "none";
  clearTop3Display(); // clear the top3 predictions

  if (currentRound === maxRounds) {
    const reloadBtn = document.getElementById("reloadButton");
    reloadBtn.innerText = "Finish";
  }
};

const endGame = () => {
  document.getElementById("drawingPage").classList.add("d-none");
  document.getElementById("resultsPage").classList.remove("d-none");

  const drawingsGrid = document.getElementById("drawingsGrid");
  roundResults.forEach((result) => {
    const imgDiv = document.createElement("div");
    imgDiv.style.width = "230px";
    imgDiv.style.height = "230px";
    imgDiv.style.border = "1px solid black";
    imgDiv.style.position = "relative";

    const img = document.createElement("img");
    img.src = result.imageData;
    img.style.width = "100%";
    img.style.height = "100%";
    imgDiv.appendChild(img);

    const label = document.createElement("div");
    label.innerText = result.isCorrect ? "Correct" : "Incorrect";
    label.style.position = "absolute";
    label.style.top = "10px";
    label.style.right = "10px";
    label.style.color = result.isCorrect ? "green" : "red";
    imgDiv.appendChild(label);

    drawingsGrid.appendChild(imgDiv);
  });
};

const clearTop3Display = () => {
  const top3Div = document.getElementById("top3");
  top3Div.innerHTML = ""; // clear the content
  while (top3Div.firstChild) {
    top3Div.removeChild(top3Div.firstChild);
  }
};

const clearResults = () => {
  const drawingsGrid = document.getElementById("drawingsGrid");
  while (drawingsGrid.firstChild) {
    drawingsGrid.removeChild(drawingsGrid.firstChild);
  }
};

const blockDrawingAndPrediction = (isCorrect = false) => {
  isBlocking = true;
  const $canvas = document.getElementById("defaultCanvas0");

  window.removeEventListener("mousedown", mouseDown);
  window.removeEventListener("mousemove", mouseMoved);
  $canvas.removeEventListener("mousedown", mouseDown);
  $canvas.removeEventListener("mousemove", mouseMoved);
  document.getElementById("clearButton").disabled = true;

  if (isCorrect) {
    score++;
    document.getElementById(
      "scoreCounter"
    ).innerText = `Score: ${score}/${maxRounds}`;
  }

  // Instead of showing the reload button, show a "next" button
  const reloadBtn = document.getElementById("reloadButton");
  reloadBtn.innerText = "Next";
  reloadBtn.style.display = "block";
  reloadBtn.onclick = resetApplication; // Add this line

  let canvas = document.getElementById("defaultCanvas0");
  roundResults.push({
    isCorrect,
    imageData: canvas.toDataURL(),
  });

  currentRound++;
};

const setRandomLabel = () => {
  const labelDiv = document.getElementById("randomLabel");
  const randomLabel = LABELS[Math.floor(Math.random() * LABELS.length)];
  labelDiv.innerText = randomLabel;
};

function inRange(n, from, to) {
  return n >= from && n < to;
}

function setup() {
  createCanvas(WIDTH, HEIGHT);
  strokeWeight(STROKE_WEIGHT);
  stroke("black");
  background("#FFFFFF");
}

function mouseDown() {
  if (isBlocking) return;

  clicked = true;
  mousePosition = [mouseX, mouseY];
}

function mouseMoved() {
  // Check whether mouse position is within canvas

  if (isBlocking) return;

  if (clicked && inRange(mouseX, 0, WIDTH) && inRange(mouseY, 0, HEIGHT)) {
    strokePixels[0].push(Math.floor(mouseX));
    strokePixels[1].push(Math.floor(mouseY));

    line(mouseX, mouseY, mousePosition[0], mousePosition[1]);
    mousePosition = [mouseX, mouseY];
  }
}

function mouseReleased() {
  if (strokePixels[0].length) {
    imageStrokes.push(strokePixels);
    strokePixels = [[], []];
  } else {
    clicked = false;
    clearTop3Display();
    return;
  }
  clicked = false;
  predict();
}

const loadModel = async () => {
  model = await tflite.loadTFLiteModel("./models/model.tflite");
  model.predict(tf.zeros([1, 28, 28, 1])); // warmup

  console.log(`Model loaded! (${LABELS.length} classes)`);
};

const preprocess = async (cb) => {
  const { min, max } = getBoundingBox();

  // Resize to 28x28 pixel & crop
  const imageBlob = await fetch("http://127.0.0.1:8000/transform", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    // redirect: "follow",
    // referrerPolicy: "no-referrer",
    body: JSON.stringify({
      strokes: imageStrokes,
      box: [min.x, min.y, max.x, max.y],
    }),
  }).then((response) => response.blob());

  const img = new Image(28, 28);
  img.src = URL.createObjectURL(imageBlob);

  img.onload = () => {
    const tensor = tf.tidy(() =>
      tf.browser.fromPixels(img, 1).toFloat().expandDims(0)
    );
    cb(tensor);
  };
};

const getMinimumCoordinates = () => {
  let min_x = Number.MAX_SAFE_INTEGER;
  let min_y = Number.MAX_SAFE_INTEGER;

  for (const stroke of imageStrokes) {
    for (let i = 0; i < stroke[0].length; i++) {
      min_x = Math.min(min_x, stroke[0][i]);
      min_y = Math.min(min_y, stroke[1][i]);
    }
  }

  return [Math.max(0, min_x), Math.max(0, min_y)];
};

const getBoundingBox = () => {
  repositionImage();

  const coords_x = [];
  const coords_y = [];

  for (const stroke of imageStrokes) {
    for (let i = 0; i < stroke[0].length; i++) {
      coords_x.push(stroke[0][i]);
      coords_y.push(stroke[1][i]);
    }
  }

  const x_min = Math.min(...coords_x);
  const x_max = Math.max(...coords_x);
  const y_min = Math.min(...coords_y);
  const y_max = Math.max(...coords_y);

  // New width & height of cropped image
  const width = Math.max(...coords_x) - Math.min(...coords_x);
  const height = Math.max(...coords_y) - Math.min(...coords_y);

  const coords_min = {
    x: Math.max(0, x_min - CROP_PADDING), // Link Kante anlegen
    y: Math.max(0, y_min - CROP_PADDING), // Obere Kante anlegen
  };
  let coords_max;

  if (width > height)
    // Left + right edge as boundary
    coords_max = {
      x: Math.min(WIDTH, x_max + CROP_PADDING), // Right edge
      y: Math.max(0, y_min + CROP_PADDING) + width, // Lower edge
    };
  // Upper + lower edge as boundary
  else
    coords_max = {
      x: Math.max(0, x_min + CROP_PADDING) + height, // Right edge
      y: Math.min(HEIGHT, y_max + CROP_PADDING), // Lower edge
    };

  return {
    min: coords_min,
    max: coords_max,
  };
};

// Reposition image to top left corner
const repositionImage = () => {
  const [min_x, min_y] = getMinimumCoordinates();
  for (const stroke of imageStrokes) {
    for (let i = 0; i < stroke[0].length; i++) {
      stroke[0][i] = stroke[0][i] - min_x + REPOS_PADDING;
      stroke[1][i] = stroke[1][i] - min_y + REPOS_PADDING;
    }
  }
};

const updateTop3Display = (top3) => {
  const top3Div = document.getElementById("top3");
  const content = `I guess it should be <strong>${
    LABELS[top3[0].index]
  }</strong>, ${LABELS[top3[1].index]}, or ${LABELS[top3[2].index]}`;
  top3Div.innerHTML = content;
};

const predict = async () => {
  if (!imageStrokes.length) return;
  if (!LABELS.length) throw new Error("No labels found!");

  preprocess((tensor) => {
    const predictions = model.predict(tensor).dataSync();

    const top3 = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: LABELS[i],
        index: i,
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 3);

    updateTop3Display(top3);

    const randomLabel = document.getElementById("randomLabel").innerText;
    if (top3.some((prediction) => prediction.className === randomLabel)) {
      clearInterval(intervalId); // Stop the timer
      blockDrawingAndPrediction(true);
    }
  });
};

const clearCanvas = () => {
  clear();
  if (pieChart) pieChart.destroy();
  background("#FFFFFF");
  imageStrokes = [];
  strokePixels = [[], []];
  clearTop3Display(); // clear the top3 predictions
};

const handleStartButton = () => {
  document.getElementById("welcomePage").classList.remove("d-flex");
  document.getElementById("welcomePage").classList.add("d-none"); // hide the welcome page
  document.getElementById("drawingPage").classList.remove("d-none"); // show the drawing page
};

window.onload = () => {
  const $clear = document.getElementById("clearButton"); // Cập nhật ID này
  const $reload = document.getElementById("reloadButton");
  const $canvas = document.getElementById("defaultCanvas0");

  loadModel();
  $canvas.addEventListener("mousedown", (e) => mouseDown(e));
  $canvas.addEventListener("mousemove", (e) => mouseMoved(e));

  $clear.addEventListener("click", (event) => {
    event.preventDefault();
    clearCanvas();
  });

  $reload.addEventListener("click", (event) => {
    event.preventDefault();
  });

  document.getElementById("startButton").addEventListener("click", () => {
    handleStartButton();
    resetApplication();
  });

  document.getElementById("restartButton").addEventListener("click", () => {
    document.getElementById("resultsPage").classList.add("d-none");
    document.getElementById("drawingPage").classList.remove("d-none");
    currentRound = 1;
    roundResults = [];
    resetApplication();
  });

  setRandomLabel();
  startTimer(timerCount);
};
