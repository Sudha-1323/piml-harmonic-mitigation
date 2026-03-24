%% ============================================================
%  PIML METHOD-A (Harmonics → THD_current using THD Equation)
%  Input  : I1, H3, H5, H7, H9, H11 magnitudes (harmonics)
%  Output : THD_current
%  Physics-informed loss enforces THD equation consistency
%
%  ✅ UPDATED VERSION:
%  - Produces Test metrics (RMSE, MAE, R2)
%  - Saves predictions for FULL dataset (1000 samples)
%  - Exports FULL predictions for BO-ANFIS
%% ============================================================

clear; clc; close all;

%% ------------------ USER SETTINGS ----------------------------
fileName = "C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\input\newinput_1000.xlsx"; % <-- change path

targetName = "THD_current";

% Required harmonic magnitude column names in your Excel
colI1  = "I1_total_A_mag";
colH3  = "H3_A_mag";
colH5  = "H5_A_mag";
colH7  = "H7_A_mag";
colH9  = "H9_A_mag";
colH11 = "H11_A_mag";

% Training parameters
trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

maxEpochs     = 60;
miniBatchSize = 64;
learnRate     = 1e-3;

% Physics loss weight
lambdaPhys = 0.8;     % increase if you want stricter physics matching
lambdaPos  = 0.1;     % positivity penalty weight (THD >= 0)

rng(1);

%% ------------------ 1) LOAD EXCEL ----------------------------
T = readtable(fileName);

disp("✅ Loaded Excel columns:");
disp(T.Properties.VariableNames);

% Check required columns exist
reqCols = [colI1 colH3 colH5 colH7 colH9 colH11 targetName];
for k = 1:numel(reqCols)
    if ~ismember(reqCols(k), string(T.Properties.VariableNames))
        error("❌ Missing column in Excel: %s", reqCols(k));
    end
end

%% ------------------ 2) EXTRACT HARMONIC FEATURES -------------
I1  = T.(colI1);
H3  = T.(colH3);
H5  = T.(colH5);
H7  = T.(colH7);
H9  = T.(colH9);
H11 = T.(colH11);

Y_raw = T.(targetName);  % measured THD_current

% Input features X = [I1 H3 H5 H7 H9 H11]
X_raw = [I1 H3 H5 H7 H9 H11];

fprintf("✅ X size = %d × %d\n", size(X_raw,1), size(X_raw,2));
fprintf("✅ Y size = %d × %d\n", size(Y_raw,1), size(Y_raw,2));

%% ------------------ 3) CLEAN NaN/Inf -------------------------
validRows = all(isfinite(X_raw),2) & isfinite(Y_raw);

X_all = X_raw(validRows,:);
Y_all = Y_raw(validRows,:);

if isempty(X_all)
    error("❌ Dataset is empty after cleaning NaN/Inf.");
end

fprintf("✅ Valid samples after cleaning: %d\n", size(X_all,1));

%% ------------------ 4) COMPUTE PHYSICS THD (Equation) --------
% THD_eq = sqrt(H3^2+H5^2+H7^2+H9^2+H11^2) / I1
I1safe_all = X_all(:,1);
I1safe_all(I1safe_all == 0) = 1e-6;

THD_eq_all = sqrt(X_all(:,2).^2 + X_all(:,3).^2 + X_all(:,4).^2 + ...
                  X_all(:,5).^2 + X_all(:,6).^2) ./ I1safe_all;

%% ------------------ 5) TRAIN/VAL/TEST SPLIT ------------------
N = size(X_all,1);
idx = randperm(N);

nTrain = floor(trainRatio*N);
nVal   = floor(valRatio*N);

idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1:nTrain+nVal);
idxTest  = idx(nTrain+nVal+1:end);

Xtrain = X_all(idxTrain,:);  Ytrain = Y_all(idxTrain);  THDtrainEq = THD_eq_all(idxTrain);
Xval   = X_all(idxVal,:);    Yval   = Y_all(idxVal);    THDvalEq   = THD_eq_all(idxVal);
Xtest  = X_all(idxTest,:);   Ytest  = Y_all(idxTest);   THDtestEq  = THD_eq_all(idxTest);

fprintf("✅ Split: Train=%d | Val=%d | Test=%d\n", size(Xtrain,1), size(Xval,1), size(Xtest,1));

%% ------------------ 6) NORMALIZE USING TRAIN ONLY ------------
muX  = mean(Xtrain,1);
sigX = std(Xtrain,0,1);
sigX(sigX==0) = 1;

XtrainN = (Xtrain - muX) ./ sigX;
XvalN   = (Xval   - muX) ./ sigX;
XtestN  = (Xtest  - muX) ./ sigX;

muY  = mean(Ytrain);
sigY = std(Ytrain);
if sigY==0, sigY=1; end

YtrainN = (Ytrain - muY) ./ sigY;
YvalN   = (Yval   - muY) ./ sigY;

% Physics THD normalization (use same Y scaling)
THDtrainEqN = (THDtrainEq - muY) ./ sigY;
THDvalEqN   = (THDvalEq   - muY) ./ sigY;

%% ------------------ 7) dlarray FORMAT ------------------------
XvalDL = dlarray(single(XvalN'), "CB");
YvalDL = dlarray(single(YvalN'), "CB");
THDvalEqDL = dlarray(single(THDvalEqN'), "CB");

XtestDL = dlarray(single(XtestN'), "CB");

%% ------------------ 8) BUILD NETWORK -------------------------
F = size(XtrainN,2);

layers = [
    featureInputLayer(F,"Name","input")
    fullyConnectedLayer(64,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(32,"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(1,"Name","out")   % output THD_current (normalized)
];

dlnet = dlnetwork(layerGraph(layers));

%% ------------------ 9) TRAINING LOOP -------------------------
trailingAvg = [];
trailingAvgSq = [];

trainLossHist = zeros(maxEpochs,1);
valLossHist   = zeros(maxEpochs,1);

numTrain = size(XtrainN,1);
iteration = 0;

disp("🚀 Training started...");

for epoch = 1:maxEpochs
    shuf = randperm(numTrain);

    epochLoss = 0;
    numB = 0;

    for i = 1:miniBatchSize:numTrain
        iteration = iteration + 1;

        batchIdx = shuf(i:min(i+miniBatchSize-1, numTrain));

        Xb = dlarray(single(XtrainN(batchIdx,:)'), "CB");
        Yb = dlarray(single(YtrainN(batchIdx,:)'), "CB");
        THDeqb = dlarray(single(THDtrainEqN(batchIdx,:)'), "CB");

        [loss, grads] = dlfeval(@modelGradients, dlnet, Xb, Yb, THDeqb, lambdaPhys, lambdaPos);

        [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, grads, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);

        epochLoss = epochLoss + double(gather(extractdata(loss)));
        numB = numB + 1;
    end

    trainLossHist(epoch) = epochLoss / numB;

    valLossHist(epoch) = computeLoss(dlnet, XvalDL, YvalDL, THDvalEqDL, lambdaPhys, lambdaPos);

    fprintf("Epoch %d/%d | TrainLoss=%.6f | ValLoss=%.6f\n", ...
        epoch, maxEpochs, trainLossHist(epoch), valLossHist(epoch));
end

disp("✅ Training finished!");

%% ------------------ 10) TEST PREDICTION ----------------------
YPredTestN = predict(dlnet, XtestDL);
YPredTestN = gather(extractdata(YPredTestN))';

% Denormalize
YPredTest = YPredTestN .* sigY + muY;

%% ------------------ 11) METRICS ------------------------------
RMSE = sqrt(mean((Ytest - YPredTest).^2));
MAE  = mean(abs(Ytest - YPredTest));
R2   = 1 - sum((Ytest-YPredTest).^2) / sum((Ytest-mean(Ytest)).^2 + 1e-12);

fprintf("\n✅ TEST PERFORMANCE:\n");
fprintf("RMSE = %.6f\n", RMSE);
fprintf("MAE  = %.6f\n", MAE);
fprintf("R2   = %.6f\n", R2);

%% ------------------ 12) FULL DATASET PREDICTION (1000 rows) ----------------
XallN = (X_all - muX) ./ sigX;
XallDL = dlarray(single(XallN'), "CB");

YPredAllN = predict(dlnet, XallDL);
YPredAllN = gather(extractdata(YPredAllN))';

% Denormalize full predicted THD
YPredAll = YPredAllN .* sigY + muY;

%% ------------------ 13) SAVE OUTPUTS --------------------------
% 13A) Save FULL dataset predictions (1000 rows) -> BEST for BO-ANFIS
outFileFull = "Predicted_THD_PIML_PhysicsEquation_FULL.xlsx";

ToutFull = table(Y_all, YPredAll, THD_eq_all, ...
    'VariableNames', {'THD_True','THD_Predicted','THD_Equation'});

writetable(ToutFull, outFileFull);
fprintf("✅ Saved FULL output (1000 rows): %s\n", outFileFull);

% 13B) Optional: Save only test-set predictions (150 rows)
outFileTest = "Predicted_THD_PIML_PhysicsEquation_TEST.xlsx";
ToutTest = table(Ytest, YPredTest, THDtestEq, ...
    'VariableNames', {'THD_True','THD_Predicted','THD_Equation'});

writetable(ToutTest, outFileTest);
fprintf("✅ Saved TEST output (150 rows): %s\n", outFileTest);

%% ------------------ 14) PLOTS --------------------------------
figure('Color','w');
plot(trainLossHist,'-o','LineWidth',1.4); hold on;
plot(valLossHist,'--s','LineWidth',1.4);
grid on;
xlabel("Epoch"); ylabel("Loss");
title("Training Progress (Harmonics → THD Physics-Informed)");
legend("Train","Val","Location","best");

figure('Color','w');
plot(Ytest,'b','LineWidth',1.2); hold on;
plot(YPredTest,'r--','LineWidth',1.2);
grid on;
xlabel("Test Sample Index"); ylabel("THD_{current}");
title("Measured vs Predicted THD_{current} (Test Set)");
legend("Measured","Predicted","Location","best");

figure('Color','w');
plot(Ytest,'b','LineWidth',1.2); hold on;
plot(YPredTest,'r--','LineWidth',1.2);
ylim([0 0.20]);
grid on;
xlabel("Test Sample Index"); ylabel("THD_{current}");
title("Zoomed THD_{current} (0–0.20) (Test Set)");
legend("Measured","Predicted","Location","best");

figure('Color','w');
plot(Ytest,'b','LineWidth',1.2); hold on;
plot(THDtestEq,'k:','LineWidth',1.5);
plot(YPredTest,'r--','LineWidth',1.2);
grid on;
xlabel("Test Sample Index"); ylabel("THD_{current}");
title("Measured vs Predicted vs Equation-based THD (Test Set)");
legend("Measured","THD Equation","Predicted","Location","best");

%% =============================================================
%                   LOCAL FUNCTIONS
%% =============================================================

function [loss, gradients] = modelGradients(net, X, Y, THDeq, lambdaPhys, lambdaPos)

    Yhat = forward(net, X);

    % Data loss (MSE)
    dataLoss = mean((Yhat - Y).^2, "all");

    % Physics loss: predicted THD should match THD from equation
    physLoss = mean((Yhat - THDeq).^2, "all");

    % Positivity loss: THD >= 0
    posLoss = mean(relu(-Yhat).^2, "all");

    loss = dataLoss + lambdaPhys*physLoss + lambdaPos*posLoss;

    gradients = dlgradient(loss, net.Learnables);
end

function L = computeLoss(net, X, Y, THDeq, lambdaPhys, lambdaPos)

    Yhat = forward(net, X);

    dataLoss = mean((Yhat - Y).^2, "all");
    physLoss = mean((Yhat - THDeq).^2, "all");
    posLoss  = mean(relu(-Yhat).^2, "all");

    L = double(gather(extractdata(dataLoss + lambdaPhys*physLoss + lambdaPos*posLoss)));
end
