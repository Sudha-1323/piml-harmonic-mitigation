function BONFISfilterctrl_twoExcel_THDC_predTarget_v3
% -------------------------------------------------------------
% BO-ANFIS Adaptive Filter Control using TWO Excel Files:
% 1) Original dataset (inputs + measured THD)
% 2) PIML FULL output file (Predicted THD: 1000 rows)
%
% Target used for BO-ANFIS = THD_current_predicted (from PIML)
% Predictors used         = P_total_W, Q_total_var, THD_voltage, microclimate_factor
%
% ✅ Publication Ready:
% - Proper 1000-row merging
% - Generates readable rules + grouped rules + compact rules
% - Generates segmented control-surface (OFF/APF/HYBRID)
% -------------------------------------------------------------

clear; clc; close all;
rng("default");

%% ================= USER SETTINGS ============================
originalFile  = "C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\input\newinput_1000.xlsx";

% ✅ IMPORTANT: Use FULL PIML output (1000 rows)
predFile      = "C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\input\Predicted_THD_PIML_PhysicsEquation_FULL.xlsx";

outputFolder  = "C:\Users\Suganya\Desktop\Harmonics-Sudha MAM\MATLAB\ANFIS_Results_TwoExcel_v3";
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end

predictorList = {'P_total_W','Q_total_var','THD_voltage','microclimate_factor'};

% Expected predicted column names inside PIML file
predictedColCandidates = ["THD_Predicted","Predicted_THD","THD_current_predicted"];

%% ================= 1) LOAD ORIGINAL DATA ====================
T = readtable(originalFile);
disp("✅ Original Excel loaded.");
fprintf("Rows: %d | Cols: %d\n", height(T), width(T));

%% ================= 2) LOAD PREDICTED THD FILE ===============
Tp = readtable(predFile);
disp("✅ Predicted THD file loaded.");
fprintf("Rows: %d | Cols: %d\n", height(Tp), width(Tp));
disp("Predicted file columns:");
disp(Tp.Properties.VariableNames);

% Find predicted THD column name
predName = "";
for k = 1:numel(predictedColCandidates)
    if any(strcmp(Tp.Properties.VariableNames, predictedColCandidates(k)))
        predName = predictedColCandidates(k);
        break;
    end
end

if predName == ""
    error("❌ Could not find predicted THD column in predicted Excel file. Rename column as THD_Predicted.");
end

THD_pred = Tp.(predName);

%% ================= 3) LENGTH CHECK ==========================
N1 = height(T);
N2 = length(THD_pred);

if N1 ~= N2
    error("❌ Row mismatch: Original has %d rows, Predicted has %d rows. Use FULL PIML output file.", N1, N2);
end

% Merge predicted THD into original dataset
T.THD_current_predicted = THD_pred;
disp("✅ Merged predicted THD into original dataset as 'THD_current_predicted'.");

%% ================= 4) CHECK PREDICTORS ======================
varNames = T.Properties.VariableNames;
missingPred = setdiff(predictorList, varNames);

if ~isempty(missingPred)
    error("❌ Missing predictors in original file: %s", strjoin(missingPred,", "));
end

%% ================= 5) BUILD INPUTS X AND TARGET Y ===========
X_full = T{:, predictorList};
Y_full = T.THD_current_predicted;  % ✅ predicted THD is target

% Clean NaN / Inf
validRows = all(isfinite(X_full),2) & isfinite(Y_full);
X_full = X_full(validRows,:);
Y_full = Y_full(validRows);

fprintf("✅ Final usable samples for BO-ANFIS: %d\n", size(X_full,1));

%% ================= 6) TRAIN/TEST SPLIT ======================
cv = cvpartition(size(X_full,1),'HoldOut',0.25);

Xtrain = X_full(training(cv),:);
Ytrain = Y_full(training(cv));

Xtest  = X_full(test(cv),:);
Ytest  = Y_full(test(cv));

%% ================= 7) NORMALIZE (MIN-MAX) ===================
minX = min(Xtrain,[],1); maxX = max(Xtrain,[],1);
denX = maxX - minX; denX(denX==0)=1;

Xtrain_n = (Xtrain - minX) ./ denX;
Xtest_n  = (Xtest  - minX) ./ denX;

minY = min(Ytrain); maxY = max(Ytrain);
denY = maxY - minY; if denY==0, denY=1; end

Ytrain_n = (Ytrain - minY) ./ denY;
Ytest_n  = (Ytest  - minY) ./ denY;

%% ================= 8) INITIAL FIS ===========================
fprintf("Generating initial FIS...\n");

try
    optGen = genfisOptions('SubtractiveClustering');
    optGen.ClusterInfluenceRange = 0.35;
    fisInit = genfis(Xtrain_n, Ytrain_n, optGen);
catch
    fisInit = genfis(Xtrain_n, Ytrain_n, 'Radius', 0.35);
end

fprintf("✅ Initial FIS: Inputs=%d, Rules=%d\n", length(fisInit.Inputs), length(fisInit.Rules));

%% ================= 9) BAYESIAN OPT ==========================
fprintf("Running Bayesian Optimization...\n");

objFcn = @(p) anfis_opt_local(p, Xtrain_n, Ytrain_n, Xtest_n, Ytest_n, fisInit);

vars = [
    optimizableVariable('InitialStepSize',[1e-3,5e-2],'Transform','log')
    optimizableVariable('EpochNumber',[30,120],'Type','integer')
];

results = bayesopt(objFcn, vars, ...
    'MaxObjectiveEvaluations', 10, ...
    'IsObjectiveDeterministic', true, ...
    'Verbose', 0);

best = results.XAtMinObjective;
fprintf("✅ Best: StepSize=%.6f, Epochs=%d\n", best.InitialStepSize, best.EpochNumber);

%% ================= 10) TRAIN FINAL ANFIS ====================
opt = anfisOptions('InitialFIS', fisInit);
opt.InitialStepSize = best.InitialStepSize;
opt.EpochNumber = best.EpochNumber;
opt.DisplayANFISInformation = 0;
opt.DisplayErrorValues = 0;

fprintf("Training final ANFIS...\n");
fisTrained_n = anfis([Xtrain_n Ytrain_n], opt);

fisFile = fullfile(outputFolder,"trained_Bayes_ANFIS_THDC_predTarget.fis");
writeFIS(fisTrained_n, fisFile);
fprintf("✅ Saved trained FIS: %s\n", fisFile);

%% ================= 11) EVALUATE TEST ========================
Ypred_n = evalfis(fisTrained_n, Xtest_n);
Ypred   = Ypred_n * denY + minY;

rmse = sqrt(mean((Ypred - Ytest).^2));
mae  = mean(abs(Ypred - Ytest));
R2   = 1 - sum((Ytest - Ypred).^2) / sum((Ytest - mean(Ytest)).^2 + 1e-12);

fprintf("\n✅ BO-ANFIS TEST RESULTS (Target = PIML Predicted THD)\n");
fprintf("RMSE = %.6f\n", rmse);
fprintf("MAE  = %.6f\n", mae);
fprintf("R2   = %.6f\n", R2);

%% ================= 12) SAVE OUTPUT ==========================
outExcel = fullfile(outputFolder,"BOANFIS_THDC_predTarget_results.xlsx");
Tout = table(Ytest, Ypred, 'VariableNames', {'THD_target_fromPIML','THD_estimated_byANFIS'});
writetable(Tout, outExcel);
fprintf("✅ Saved results Excel: %s\n", outExcel);

%% ================= 13) PLOT PERFORMANCE =====================
figure('Color','w');
plot(Ytest,'b','LineWidth',1.2); hold on;
plot(Ypred,'r--','LineWidth',1.2);
grid on;
xlabel("Test Sample Index");
ylabel("THD_{current}^{pred}");
title("BO-ANFIS Prediction (Target = PIML Predicted THD)");
legend("Target (PIML)","BO-ANFIS Output","Location","best");

%% ================= 14) CREATE READABLE RULES =================
ruleFileRaw      = fullfile(outputFolder,"BOANFIS_rules_raw.txt");
ruleFileReadable = fullfile(outputFolder,"BOANFIS_rules_readable.txt");
ruleFileGrouped  = fullfile(outputFolder,"BOANFIS_rules_grouped.txt");
ruleFileCompact  = fullfile(outputFolder,"BOANFIS_rules_compact.txt");

% Raw showrule output
sr = showrule(fisTrained_n);
writelines(string(sr), ruleFileRaw);

% Basic readable rules (text conversion)
readableRules = make_readable_rules(fisTrained_n, sr, predictorList);
writelines(readableRules, ruleFileReadable);

% Adaptive thresholds from training predicted THD
q = quantile(Ytrain, [0, 0.33, 0.66, 1]);
th_low = q(2);
th_high = q(3);

groupedText = group_rules_by_mode(readableRules, th_low, th_high);
writelines(groupedText, ruleFileGrouped);

compactText = compact_natural_rules(th_low, th_high);
writelines(compactText, ruleFileCompact);

fprintf("✅ Rules saved:\n");
fprintf("Raw      : %s\n", ruleFileRaw);
fprintf("Readable : %s\n", ruleFileReadable);
fprintf("Grouped  : %s\n", ruleFileGrouped);
fprintf("Compact  : %s\n", ruleFileCompact);

%% ================= 15) CONTROL SURFACE (THD_pred vs microclimate) ===========
% Surface will show OFF/APF/HYBRID modes using thresholds

ix_micro = find(strcmp(predictorList,'microclimate_factor'));
if isempty(ix_micro), ix_micro = 4; end

nx = 80; ny = 80;

% THD axis range (use predicted THD train range)
xRange = linspace(min(Ytrain), max(Ytrain), nx);
yRange = linspace(min(Xtrain(:,ix_micro)), max(Xtrain(:,ix_micro)), ny);
[XX,YY] = meshgrid(xRange, yRange);

% For plotting surface, keep other predictors mean
gridX = repmat(mean(Xtrain,1), numel(XX), 1);
gridX(:,ix_micro) = YY(:);

% But THD_pred is NOT predictor here (it is target).
% So we use THD thresholds only for segmentation display.

Mode = zeros(size(XX));
Mode(XX < th_low) = 0;
Mode(XX >= th_low & XX < th_high) = 1;
Mode(XX >= th_high) = 2;

figSurf = figure('Color','w','Units','pixels','Position',[100 100 1100 750]);
surf(XX, YY, Mode, 'EdgeColor','none'); shading flat;
colormap([0.6 0.85 1; 1 1 0.6; 1 0.75 0.75]);
c = colorbar;
c.Ticks = [0,1,2];
c.TickLabels = {'OFF','APF','HYBRID'};
xlabel('THD_{current}^{pred} (PIML)');
ylabel('microclimate\_factor');
zlabel('Filter Mode (categorical)');
title('Filter Control Surface (Predicted THD vs Microclimate Factor)','FontWeight','bold');
view(135,28); grid on;

text(mean(xRange), max(yRange), 2.05, ...
    sprintf('THD_{OFF/APF}=%.4f\nTHD_{APF/HYB}=%.4f', th_low, th_high), ...
    'Color','k','FontSize',10,'BackgroundColor','w','EdgeColor','k');

pngSurf = fullfile(outputFolder,"Filter_Control_Surface_THDpred_micro_segmented.png");
print(figSurf, pngSurf, '-dpng', '-r300');
fprintf("✅ Saved control surface: %s\n", pngSurf);

fprintf("\n✅ DONE. All outputs saved to: %s\n", outputFolder);

end

%% ================= LOCAL FUNCTIONS ===========================

function obj = anfis_opt_local(p, Xt, Yt, Xv, Yv, fisInit)
    try
        optL = anfisOptions('InitialFIS', fisInit);
        optL.InitialStepSize = p.InitialStepSize;
        optL.EpochNumber = p.EpochNumber;
        optL.DisplayANFISInformation = 0;
        optL.DisplayErrorValues = 0;

        fisTmp = anfis([Xt Yt], optL);
        Ypred = evalfis(fisTmp, Xv);

        err = mean(abs(Ypred - Yv), 'omitnan'); % MAE normalized
        if ~isfinite(err), err = 1e6; end
        obj = err;
    catch
        obj = 1e6;
    end
end

function readable = make_readable_rules(fis, sr_raw, predictorNames)

    sr = string(sr_raw);
    readable = strings(0,1);

    function lbl = mfLabelFor(k)
        switch k
            case 1, lbl = "Low";
            case 2, lbl = "Medium";
            case 3, lbl = "High";
            case 4, lbl = "VeryHigh";
            otherwise, lbl = "MF" + string(k);
        end
    end

    for r = 1:numel(sr)
        s0 = char(sr(r));
        expr = 'in(\d+)cluster(\d+)';
        tokens = regexp(s0, expr, 'tokens');

        if ~isempty(tokens)
            for t = 1:numel(tokens)
                idxNum = str2double(tokens{t}{1});
                clNum  = str2double(tokens{t}{2});

                if idxNum>=1 && idxNum <= numel(predictorNames)
                    pname = predictorNames{idxNum};
                else
                    pname = sprintf('Var%d', idxNum);
                end

                label = char(mfLabelFor(clNum));
                repl = sprintf('(%s is %s)', pname, label);
                s0 = regexprep(s0, sprintf('in%dcluster%d', idxNum, clNum), repl);
            end
        end

        s0 = regexprep(s0,'\bif\b','IF','ignorecase');
        s0 = regexprep(s0,'\bthen\b','THEN','ignorecase');
        s0 = strrep(s0,'&','AND');
        s0 = regexprep(s0,'\s+',' ');

        readable(end+1) = string(s0);
    end
end

function groupedText = group_rules_by_mode(readableRules, th_low, th_high)

    header = [
        "Grouped BO-ANFIS Rules (Target = Predicted THD_current)";
        "=======================================================";
        "";
        sprintf("Adaptive thresholds (based on predicted THD_current quantiles):");
        sprintf("  THD < %.4f  -> OFF", th_low);
        sprintf("  %.4f ≤ THD < %.4f  -> APF", th_low, th_high);
        sprintf("  THD ≥ %.4f  -> HYBRID", th_high);
        "";
        '--- FILTER OFF RULES ---';
        ""
        ];

    offRules = readableRules(contains(readableRules,'out1cluster1','IgnoreCase',true));
    apfRules = readableRules(contains(readableRules,'out1cluster2','IgnoreCase',true));
    hybRules = readableRules(contains(readableRules,'out1cluster3','IgnoreCase',true));

    if isempty(offRules), offRules = "(none found)"; end
    if isempty(apfRules), apfRules = "(none found)"; end
    if isempty(hybRules), hybRules = "(none found)"; end

    groupedText = [header; string(offRules); ""; '--- FILTER APF RULES ---'; ""; string(apfRules); ""; '--- FILTER HYBRID RULES ---'; ""; string(hybRules)];
end

function compactText = compact_natural_rules(th_low, th_high)

    compactText = [
        "BO-ANFIS — Compact filter activation rules (numeric thresholds)";
        "";
        sprintf("THD_current_pred < %.4f  -> OFF", th_low);
        sprintf("%.4f ≤ THD_current_pred < %.4f -> APF", th_low, th_high);
        sprintf("THD_current_pred ≥ %.4f -> HYBRID", th_high);
        "";
        "Compact rules:";
        sprintf("1) IF THD_current_pred < %.4f THEN Filter = OFF", th_low);
        sprintf("2) IF THD_current_pred >= %.4f AND THD_current_pred < %.4f THEN Filter = APF", th_low, th_high);
        sprintf("3) IF THD_current_pred >= %.4f THEN Filter = HYBRID", th_high);
        "";
        "Note: microclimate_factor can be added as a secondary refinement variable."
        ];
end
