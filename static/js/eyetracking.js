/**
 * Gazer - Eye Tracking Frontend
 * 
 * Real-time eye tracking using MediaPipe Face Landmarker and WebSocket
 * communication with the Flask backend for gaze prediction.
 * 
 * @author Your Name
 * @version 1.0.0
 */

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    // Calibration settings
    POINTS_PER_CALIBRATION: 100,
    POINTS_PER_VALIDATION: 300,
    POINTS_PER_TEST: 400,
    
    // Display settings
    VIDEO_WIDTH: 480,
    CALIBRATION_POINT_RADIUS: 35,
    PREDICTION_POINT_RADIUS: 50,
    
    // Blink detection threshold
    BLINK_THRESHOLD: 0.25,
    
    // MediaPipe model URL
    MODEL_URL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    
    // Colors
    COLORS: {
        LEFT_IRIS: "#30FF30",
        RIGHT_IRIS: "#FF3030",
        CALIBRATION_FILL: "black",
        CALIBRATION_STROKE: "white",
        VALIDATION_STROKE: "red",
        PREDICTION: "white",
        BACKGROUND: "#5e6c7b",
    }
};

// ============================================================================
// State Management
// ============================================================================

const state = {
    // MediaPipe
    faceLandmarker: null,
    runningMode: "IMAGE",
    
    // Webcam
    webcamRunning: false,
    lastVideoTime: -1,
    
    // Tracking data
    leftPupil: { x: 0, y: 0 },
    rightPupil: { x: 0, y: 0 },
    nose: { x: 0, y: 0 },
    blinkScores: { left: 0, right: 0 },
    
    // Predictions
    predictedX: 0,
    predictedY: 0,
    
    // Calibration state
    calibrationDone: false,
    modelTrained: false,
    validationProcess: false,
    validationDone: false,
    testing: false,
    
    // Counters
    pointCounter: 0,
    validationPointer: 0,
    testPointer: 0,
    
    // Test metrics
    testDiffX: 0,
    testDiffY: 0,
    testX: 0,
    testY: 0,
};

// ============================================================================
// DOM Elements
// ============================================================================

const elements = {
    video: document.getElementById("webcam"),
    canvas: document.getElementById("output_canvas"),
    calibrationCanvas: document.getElementById("calibrationCanvas"),
    calibrationButton: document.getElementById("calibrationButton"),
    webcamButton: document.getElementById("webcamButton"),
    demosSection: document.getElementById("demos"),
    videoBlendShapes: document.getElementById("video-blend-shapes"),
    heading: document.getElementById("heading"),
    body: document.getElementById("body"),
    loading: document.getElementById("loading"),
};

// Get canvas contexts
const canvasCtx = elements.canvas.getContext("2d");
const calibrationCtx = elements.calibrationCanvas.getContext("2d");

// Hide loading initially
elements.loading.style.display = "none";
elements.calibrationCanvas.style.display = "none";

// ============================================================================
// MediaPipe Setup
// ============================================================================

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
const drawingUtils = new DrawingUtils(canvasCtx);

/**
 * Initialize the MediaPipe Face Landmarker.
 */
async function initializeFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    
    state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: CONFIG.MODEL_URL,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: state.runningMode,
        numFaces: 1
    });
    
    elements.demosSection.classList.remove("invisible");
    console.log("FaceLandmarker initialized");
}

// ============================================================================
// Socket.IO Setup
// ============================================================================

const socket = io();

socket.on("connect", () => {
    console.log("Connected to server");
});

socket.on("disconnect", () => {
    console.log("Disconnected from server");
});

socket.on("modelTrained", () => {
    console.log("Model training complete");
    state.modelTrained = true;
    state.validationProcess = true;
    elements.loading.remove();
    startValidation();
});

socket.on("validationStatus", () => {
    console.log("Validation complete");
    state.validationDone = true;
    state.testing = true;
    startAccuracyTest();
});

socket.on("data_response", (data) => {
    const [x, y] = JSON.parse(data);
    state.predictedX = x;
    state.predictedY = y;
    
    // Draw prediction point if not in validation or testing
    if (!state.validationProcess && !state.testing) {
        drawPredictionPoint(x, y);
    }
});

// ============================================================================
// Webcam Functions
// ============================================================================

/**
 * Check if webcam access is supported.
 */
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Enable or disable webcam feed.
 */
async function toggleWebcam() {
    if (!state.faceLandmarker) {
        console.log("FaceLandmarker not loaded yet");
        return;
    }
    
    if (state.webcamRunning) {
        state.webcamRunning = false;
        elements.webcamButton.innerText = "ENABLE WEBCAM";
        elements.calibrationButton.style.display = "none";
    } else {
        state.webcamRunning = true;
        elements.webcamButton.innerText = "DISABLE WEBCAM";
        elements.calibrationButton.style.display = "inline-block";
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });
        
        elements.video.srcObject = stream;
        elements.video.addEventListener("loadeddata", processVideoFrame);
    }
}

/**
 * Process each video frame for face landmark detection.
 */
async function processVideoFrame() {
    // Set video dimensions
    const ratio = elements.video.videoHeight / elements.video.videoWidth;
    elements.video.style.width = `${CONFIG.VIDEO_WIDTH}px`;
    elements.video.style.height = `${CONFIG.VIDEO_WIDTH * ratio}px`;
    elements.canvas.style.width = `${CONFIG.VIDEO_WIDTH}px`;
    elements.canvas.style.height = `${CONFIG.VIDEO_WIDTH * ratio}px`;
    elements.canvas.width = elements.video.videoWidth;
    elements.canvas.height = elements.video.videoHeight;
    
    // Switch to video mode if needed
    if (state.runningMode === "IMAGE") {
        state.runningMode = "VIDEO";
        await state.faceLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    
    // Process frame if new
    if (state.lastVideoTime !== elements.video.currentTime) {
        state.lastVideoTime = elements.video.currentTime;
        const results = state.faceLandmarker.detectForVideo(
            elements.video,
            performance.now()
        );
        
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            processLandmarks(results);
        }
    }
    
    // Continue loop if webcam is running
    if (state.webcamRunning) {
        requestAnimationFrame(processVideoFrame);
    }
}

/**
 * Process detected face landmarks and extract eye/nose positions.
 */
function processLandmarks(results) {
    const landmarks = results.faceLandmarks[0];
    const canvasWidth = elements.canvas.width;
    const canvasHeight = elements.canvas.height;
    
    // Extract iris landmarks (indices 469-472 for left, 474-477 for right)
    const leftIris = landmarks.slice(469, 473);
    const rightIris = landmarks.slice(474, 478);
    
    // Calculate pupil centers
    state.leftPupil = {
        x: ((leftIris[0].x + leftIris[2].x) / 2) * canvasWidth,
        y: ((leftIris[1].y + leftIris[3].y) / 2) * canvasHeight
    };
    
    state.rightPupil = {
        x: ((rightIris[0].x + rightIris[2].x) / 2) * canvasWidth,
        y: ((rightIris[1].y + rightIris[3].y) / 2) * canvasHeight
    };
    
    // Nose tip (landmark index 1)
    state.nose = {
        x: landmarks[1].x * canvasWidth,
        y: landmarks[1].y * canvasHeight
    };
    
    // Blink detection (blend shape indices 9 and 10)
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        state.blinkScores = {
            left: results.faceBlendshapes[0].categories[9].score,
            right: results.faceBlendshapes[0].categories[10].score
        };
    }
    
    // Draw iris landmarks
    drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: CONFIG.COLORS.RIGHT_IRIS }
    );
    drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: CONFIG.COLORS.LEFT_IRIS }
    );
    
    // Send real-time data if calibrated and trained
    if (state.calibrationDone && state.modelTrained) {
        sendTrackingData();
    }
    
    // Update blend shapes display
    updateBlendShapesDisplay(results.faceBlendshapes);
}

/**
 * Send current tracking data to the server.
 */
function sendTrackingData() {
    const data = {
        leftX: state.leftPupil.x,
        leftY: state.leftPupil.y,
        rightX: state.rightPupil.x,
        rightY: state.rightPupil.y,
        noseX: state.nose.x,
        noseY: state.nose.y,
        blink: isBlinking() ? 1 : 0,
        validationDone: state.validationDone
    };
    
    socket.emit("realTimeData", JSON.stringify(data));
}

/**
 * Check if user is currently blinking.
 */
function isBlinking() {
    return state.blinkScores.left >= CONFIG.BLINK_THRESHOLD ||
           state.blinkScores.right >= CONFIG.BLINK_THRESHOLD;
}

// ============================================================================
// Calibration Functions
// ============================================================================

/**
 * Generate calibration point coordinates.
 * Returns 21 points covering the screen systematically.
 */
function generateCalibrationPoints(width, height) {
    const positions = [
        [0.5, 0.5],   // center
        [0.05, 0.05], // top-left
        [0.25, 0.05],
        [0.5, 0.05],  // top-center
        [0.75, 0.05],
        [0.95, 0.05], // top-right
        [0.25, 0.25], // 2nd quadrant
        [0.5, 0.25],
        [0.75, 0.25], // 1st quadrant
        [0.05, 0.5],  // left
        [0.25, 0.5],
        [0.75, 0.5],
        [0.95, 0.5],  // right
        [0.25, 0.75], // 3rd quadrant
        [0.5, 0.75],
        [0.75, 0.75], // 4th quadrant
        [0.05, 0.95], // bottom-left
        [0.25, 0.95],
        [0.5, 0.95],  // bottom-center
        [0.75, 0.95],
        [0.95, 0.95], // bottom-right
    ];
    
    return positions.map(([xRatio, yRatio]) => ({
        x: xRatio * width,
        y: yRatio * height
    }));
}

/**
 * Generate validation point coordinates.
 */
function generateValidationPoints(width, height) {
    const positions = [
        [0.5, 0.5],   // center
        [0.25, 0.25], // 2nd quad
        [0.05, 0.05], // top-left
        [0.75, 0.25], // 1st quad
        [0.95, 0.05], // top-right
        [0.25, 0.75], // 3rd quad
        [0.05, 0.95], // bottom-left
        [0.75, 0.75], // 4th quad
        [0.95, 0.95], // bottom-right
    ];
    
    return positions.map(([xRatio, yRatio]) => ({
        x: xRatio * width,
        y: yRatio * height
    }));
}

/**
 * Start the calibration process.
 */
function startCalibration() {
    // Setup canvas
    elements.calibrationCanvas.style.display = "block";
    elements.calibrationCanvas.style.backgroundColor = CONFIG.COLORS.BACKGROUND;
    elements.calibrationCanvas.width = window.innerWidth;
    elements.calibrationCanvas.height = window.innerHeight;
    
    calibrationCtx.fillStyle = "black";
    calibrationCtx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    
    const points = generateCalibrationPoints(
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    let currentPointIndex = 0;
    state.pointCounter = 0;
    
    function processCalibrationFrame() {
        if (currentPointIndex >= points.length) {
            finishCalibration();
            return;
        }
        
        const point = points[currentPointIndex];
        
        if (state.pointCounter >= CONFIG.POINTS_PER_CALIBRATION) {
            currentPointIndex++;
            state.pointCounter = 0;
        } else {
            // Draw shrinking calibration point
            const progress = state.pointCounter / CONFIG.POINTS_PER_CALIBRATION;
            drawCalibrationPoint(point.x, point.y, 1 - progress);
            
            // Send calibration data
            const data = {
                screenX: point.x,
                screenY: point.y,
                leftX: state.leftPupil.x,
                leftY: state.leftPupil.y,
                rightX: state.rightPupil.x,
                rightY: state.rightPupil.y,
                noseX: state.nose.x,
                noseY: state.nose.y,
                blink: isBlinking() ? 1 : 0
            };
            
            socket.emit("calibrationDataOneByOneUpdate", JSON.stringify(data));
            state.pointCounter++;
        }
        
        requestAnimationFrame(processCalibrationFrame);
    }
    
    processCalibrationFrame();
}

/**
 * Draw a calibration point with given radius multiplier.
 */
function drawCalibrationPoint(x, y, radiusMultiplier) {
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    // Outer ring
    calibrationCtx.strokeStyle = CONFIG.COLORS.CALIBRATION_STROKE;
    calibrationCtx.beginPath();
    calibrationCtx.arc(x, y, CONFIG.CALIBRATION_POINT_RADIUS, 0, Math.PI * 2);
    calibrationCtx.stroke();
    
    // Inner filled circle (shrinking)
    calibrationCtx.fillStyle = CONFIG.COLORS.CALIBRATION_FILL;
    calibrationCtx.beginPath();
    calibrationCtx.arc(
        x, y,
        CONFIG.CALIBRATION_POINT_RADIUS * radiusMultiplier,
        0, Math.PI * 2
    );
    calibrationCtx.fill();
}

/**
 * Draw prediction point.
 */
function drawPredictionPoint(x, y) {
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    calibrationCtx.fillStyle = CONFIG.COLORS.PREDICTION;
    calibrationCtx.beginPath();
    calibrationCtx.arc(x, y, CONFIG.PREDICTION_POINT_RADIUS, 0, Math.PI * 2);
    calibrationCtx.fill();
}

/**
 * Finish calibration and trigger model training.
 */
function finishCalibration() {
    state.calibrationDone = true;
    elements.loading.style.display = "block";
    socket.emit("calibrationStatus", true);
    
    if (elements.heading) {
        elements.heading.remove();
    }
    
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    console.log("Calibration complete, waiting for model training...");
}

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Start the validation process.
 */
function startValidation() {
    console.log("Starting validation process");
    
    const points = generateValidationPoints(
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    let currentPointIndex = 0;
    state.validationPointer = 0;
    
    function processValidationFrame() {
        if (currentPointIndex >= points.length) {
            finishValidation();
            return;
        }
        
        const point = points[currentPointIndex];
        
        if (state.validationPointer >= CONFIG.POINTS_PER_VALIDATION) {
            currentPointIndex++;
            state.validationPointer = 0;
        } else {
            // Draw validation point and prediction
            drawValidationPoint(point.x, point.y, state.predictedX, state.predictedY);
            
            // Send validation data (filtered by distance)
            const data = {
                screen_x: point.x,
                screen_y: point.y,
                predicted_x: state.predictedX,
                predicted_y: state.predictedY
            };
            
            socket.emit("validationData", JSON.stringify(data));
            state.validationPointer++;
        }
        
        requestAnimationFrame(processValidationFrame);
    }
    
    processValidationFrame();
}

/**
 * Draw validation point with prediction indicator.
 */
function drawValidationPoint(targetX, targetY, predictedX, predictedY) {
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    // Target point (red outline)
    calibrationCtx.strokeStyle = CONFIG.COLORS.VALIDATION_STROKE;
    calibrationCtx.fillStyle = CONFIG.COLORS.CALIBRATION_FILL;
    calibrationCtx.beginPath();
    calibrationCtx.arc(targetX, targetY, 60, 0, Math.PI * 2);
    calibrationCtx.fill();
    calibrationCtx.stroke();
    
    // Prediction point (white)
    calibrationCtx.fillStyle = CONFIG.COLORS.PREDICTION;
    calibrationCtx.beginPath();
    calibrationCtx.arc(predictedX, predictedY, CONFIG.PREDICTION_POINT_RADIUS, 0, Math.PI * 2);
    calibrationCtx.fill();
}

/**
 * Finish validation process.
 */
function finishValidation() {
    state.validationProcess = false;
    socket.emit("validationStatus", true);
    
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    console.log("Validation complete");
}

// ============================================================================
// Accuracy Testing
// ============================================================================

/**
 * Start accuracy testing with random points.
 */
function startAccuracyTest() {
    console.log("Starting accuracy test");
    
    state.testDiffX = 0;
    state.testDiffY = 0;
    state.testX = Math.random() * elements.calibrationCanvas.width;
    state.testY = Math.random() * elements.calibrationCanvas.height;
    state.testPointer = 0;
    
    let currentPointIndex = 0;
    const totalTestPoints = 5;
    
    function processTestFrame() {
        if (currentPointIndex >= totalTestPoints) {
            finishAccuracyTest();
            return;
        }
        
        if (state.testPointer >= CONFIG.POINTS_PER_TEST) {
            currentPointIndex++;
            state.testPointer = 0;
            state.testX = Math.random() * elements.calibrationCanvas.width;
            state.testY = Math.random() * elements.calibrationCanvas.height;
        } else {
            drawValidationPoint(state.testX, state.testY, state.predictedX, state.predictedY);
            
            // Accumulate error
            state.testDiffX += Math.abs(state.predictedX - state.testX);
            state.testDiffY += Math.abs(state.predictedY - state.testY);
            
            state.testPointer++;
        }
        
        requestAnimationFrame(processTestFrame);
    }
    
    processTestFrame();
}

/**
 * Calculate and display final accuracy.
 */
function finishAccuracyTest() {
    state.testing = false;
    
    const totalPoints = 5 * CONFIG.POINTS_PER_TEST;
    const avgError = (state.testDiffX + state.testDiffY) / (2 * totalPoints);
    const normalizedError = avgError / 50;  // Normalize to percentage
    const accuracy = Math.max(0, 100 - (normalizedError * 100));
    
    console.log("=== Accuracy Test Results ===");
    console.log(`Total X Error: ${state.testDiffX.toFixed(2)}`);
    console.log(`Total Y Error: ${state.testDiffY.toFixed(2)}`);
    console.log(`Average Error: ${avgError.toFixed(2)} pixels`);
    console.log(`Estimated Accuracy: ${accuracy.toFixed(2)}%`);
    
    // Clear canvas and show results
    calibrationCtx.clearRect(
        0, 0,
        elements.calibrationCanvas.width,
        elements.calibrationCanvas.height
    );
    
    // Display accuracy on canvas
    calibrationCtx.fillStyle = "white";
    calibrationCtx.font = "48px Arial";
    calibrationCtx.textAlign = "center";
    calibrationCtx.fillText(
        `Accuracy: ${accuracy.toFixed(1)}%`,
        elements.calibrationCanvas.width / 2,
        elements.calibrationCanvas.height / 2
    );
}

// ============================================================================
// UI Updates
// ============================================================================

/**
 * Update blend shapes display for blink detection.
 */
function updateBlendShapesDisplay(blendShapes) {
    if (!blendShapes || !blendShapes.length) return;
    
    const categories = blendShapes[0].categories;
    
    // Display left and right blink scores (indices 9 and 10)
    const html = [9, 10].map(index => {
        const shape = categories[index];
        const width = Math.min(shape.score * 100, 100);
        return `
            <li class="blend-shapes-item">
                <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
                <span class="blend-shapes-value" style="width: calc(${width}% - 120px)">
                    ${shape.score.toFixed(4)}
                </span>
            </li>
        `;
    }).join("");
    
    elements.videoBlendShapes.innerHTML = html;
}

// ============================================================================
// Event Listeners
// ============================================================================

// Window resize handler
window.addEventListener("resize", () => {
    if (elements.calibrationCanvas.style.display !== "none") {
        elements.calibrationCanvas.width = window.innerWidth;
        elements.calibrationCanvas.height = window.innerHeight;
    }
});

// Setup webcam button
if (hasGetUserMedia()) {
    elements.webcamButton.addEventListener("click", toggleWebcam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Setup calibration button
elements.calibrationButton.addEventListener("click", startCalibration);

// ============================================================================
// Initialization
// ============================================================================

initializeFaceLandmarker();
